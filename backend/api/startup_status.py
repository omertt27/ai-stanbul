"""
Startup Status Diagnostic Endpoint

Provides detailed information about the startup initialization state
"""

from fastapi import APIRouter
from core.startup_fixed import fast_startup_manager as startup_manager
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/startup", tags=["Diagnostics"])


@router.get("/status")
async def get_startup_status():
    """Get detailed startup status"""
    
    pure_llm_core = startup_manager.get_pure_llm_core()
    
    status = {
        "llm_initialized": startup_manager._llm_initialized,
        "pure_llm_core_exists": pure_llm_core is not None,
        "services_initialized": startup_manager._services_initialized,
        "service_manager_exists": startup_manager.service_manager is not None,
        "db_exists": startup_manager.db is not None,
        "redis_cache_exists": startup_manager.redis_cache is not None,
        "is_llm_ready": startup_manager.is_llm_ready(),
        "is_services_ready": startup_manager.is_services_ready(),
        "initialization_errors": getattr(startup_manager, '_initialization_errors', [])
    }
    
    if pure_llm_core:
        status["llm_client_exists"] = hasattr(pure_llm_core, 'llm_client') and pure_llm_core.llm_client is not None
    
    logger.info(f"📊 Startup status: {status}")
    
    return status
