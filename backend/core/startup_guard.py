"""
Startup Guard - Prevents duplicate initialization across multiple app instances

This is critical for FastAPI + Uvicorn with reload mode, which can trigger
initialization from multiple sources:
- Direct script execution (__main__)
- Multiprocessing (__mp_main__)
- Uvicorn reload fork (main_modular)

Author: AI Istanbul Team
Date: December 27, 2025
"""

import logging
import threading
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Thread-safe lock for initialization
_init_lock = threading.Lock()

# Global initialization flags
_initialized: Dict[str, bool] = {
    "database": False,
    "redis": False,
    "cache_warming": False,
    "models": False,
    "services": False,
    "full_startup": False,
}

# Track initialization times for debugging
_init_times: Dict[str, datetime] = {}

# Redis readiness - single source of truth
_redis_ready: Optional[bool] = None
_redis_client = None


def is_initialized(component: str) -> bool:
    """Check if a component has been initialized."""
    return _initialized.get(component, False)


def mark_initialized(component: str) -> bool:
    """
    Mark a component as initialized.
    
    Returns:
        True if this was the FIRST initialization (proceed with init)
        False if already initialized (skip init)
    """
    global _initialized, _init_times
    
    with _init_lock:
        if _initialized.get(component, False):
            logger.debug(f"🔁 {component} already initialized, skipping")
            return False
        
        _initialized[component] = True
        _init_times[component] = datetime.now()
        logger.info(f"✅ {component} marked as initialized")
        return True


def ensure_single_init(component: str = "full_startup") -> bool:
    """
    Ensure a component is only initialized once.
    
    Usage:
        if not ensure_single_init("database"):
            return  # Skip, already initialized
        
        # ... do heavy initialization ...
    
    Returns:
        True if this is the first call (should proceed)
        False if already initialized (should skip)
    """
    return mark_initialized(component)


def reset_init_state(component: Optional[str] = None):
    """
    Reset initialization state (for testing or hot reload).
    
    Args:
        component: Specific component to reset, or None to reset all
    """
    global _initialized, _init_times, _redis_ready, _redis_client
    
    with _init_lock:
        if component:
            _initialized[component] = False
            _init_times.pop(component, None)
            logger.info(f"🔄 Reset init state for: {component}")
        else:
            _initialized = {k: False for k in _initialized}
            _init_times.clear()
            _redis_ready = None
            _redis_client = None
            logger.info("🔄 Reset ALL init states")


# =============================================================================
# Redis Readiness - Single Source of Truth
# =============================================================================

def check_redis_once() -> bool:
    """
    Check Redis readiness ONCE and cache the result.
    All components should use this instead of checking independently.
    
    Returns:
        True if Redis is ready, False otherwise
    """
    global _redis_ready, _redis_client
    
    with _init_lock:
        # Return cached result if already checked
        if _redis_ready is not None:
            return _redis_ready
        
        # First time check
        try:
            import redis
            import os
            
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            client = redis.from_url(redis_url, socket_timeout=2.0, socket_connect_timeout=2.0)
            client.ping()
            
            _redis_ready = True
            _redis_client = client
            logger.info("✅ Redis readiness check: CONNECTED")
            return True
            
        except Exception as e:
            _redis_ready = False
            _redis_client = None
            logger.warning(f"⚠️ Redis readiness check: NOT AVAILABLE ({e})")
            return False


def get_redis_client():
    """
    Get the cached Redis client if available.
    Always call check_redis_once() first or use this which does it automatically.
    """
    global _redis_ready, _redis_client
    
    if _redis_ready is None:
        check_redis_once()
    
    return _redis_client if _redis_ready else None


def is_redis_ready() -> bool:
    """Check if Redis is ready (uses cached result)."""
    global _redis_ready
    
    if _redis_ready is None:
        return check_redis_once()
    
    return _redis_ready


# =============================================================================
# Initialization Status Report
# =============================================================================

def get_init_status() -> Dict:
    """Get current initialization status for debugging."""
    return {
        "initialized": dict(_initialized),
        "init_times": {k: v.isoformat() for k, v in _init_times.items()},
        "redis_ready": _redis_ready,
    }


def log_init_status():
    """Log current initialization status."""
    status = get_init_status()
    logger.info(f"📊 Init Status: {status['initialized']}")
    if status['init_times']:
        for component, time in status['init_times'].items():
            logger.info(f"   {component}: {time}")
