"""
Health Check Endpoints Module

System health monitoring and diagnostics
"""

from fastapi import APIRouter, HTTPException, status, Request
from datetime import datetime
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/health", tags=["System Health"])


@router.get("")
async def health_check():
    """
    Simple health check for Cloud Run startup probe
    
    CRITICAL: This endpoint MUST respond quickly without dependencies.
    - NO database checks
    - NO external API calls  
    - NO blocking operations
    
    For detailed health, use /api/health/detailed
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/detailed")
async def detailed_health_check():
    """
    Detailed health check with all subsystems
    
    Returns comprehensive health status including:
    - Service availability
    - Circuit breaker states
    - Performance metrics
    - Resource utilization
    - Subsystem status
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.1.0",
        "environment": "production",
        "subsystems": {}
    }
    
    # Check Pure LLM Handler
    try:
        from core.startup import startup_manager
        
        pure_llm_core = startup_manager.get_pure_llm_core()
        if pure_llm_core:
            llm_health = pure_llm_core.get_health_status()
            health_status["subsystems"]["pure_llm"] = llm_health
        else:
            health_status["subsystems"]["pure_llm"] = {
                "status": "unavailable",
                "message": "Pure LLM Handler not initialized"
            }
    except Exception as e:
        logger.error(f"Error checking Pure LLM health: {e}")
        health_status["subsystems"]["pure_llm"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Check database
    try:
        from database import engine
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        health_status["subsystems"]["database"] = {
            "status": "healthy",
            "type": "postgresql"
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        health_status["subsystems"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check ML service
    try:
        from backend.ml_service_client import check_ml_health
        ml_health = await check_ml_health()
        health_status["subsystems"]["ml_service"] = ml_health
    except Exception as e:
        logger.error(f"ML service health check failed: {e}")
        health_status["subsystems"]["ml_service"] = {
            "status": "unavailable",
            "error": str(e)
        }
    
    # Check Redis Cache
    try:
        from services.redis_cache import get_cache_service
        cache = get_cache_service()
        cache_stats = await cache.get_stats()
        health_status["subsystems"]["cache_service"] = {
            "status": "healthy" if cache_stats.get("enabled") else "degraded",
            "stats": cache_stats
        }
    except Exception as e:
        logger.error(f"Cache health check failed: {e}")
        health_status["subsystems"]["cache_service"] = {
            "status": "unavailable",
            "error": str(e)
        }
    
    return health_status


@router.get("/pure-llm")
async def pure_llm_health():
    """
    Pure LLM Handler health check
    
    Returns detailed health status including:
    - Circuit breaker states
    - Timeout metrics
    - Service availability
    - Performance metrics
    """
    from core.startup import startup_manager
    
    pure_llm_core = startup_manager.get_pure_llm_core()
    
    if not pure_llm_core:
        return {
            "status": "unavailable",
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Pure LLM Handler not initialized"
        }
    
    try:
        health_status = pure_llm_core.get_health_status()
        return {
            **health_status,
            "architecture": "modular_with_resilience",
            "modules": [
                "core", "signals", "context", "prompts", "analytics",
                "caching", "conversation", "query_enhancement", 
                "experimentation", "resilience"
            ]
        }
    except Exception as e:
        logger.error(f"Pure LLM health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check error: {str(e)}"
        )


@router.get("/circuit-breakers")
async def circuit_breakers_test():
    """
    Test all circuit breakers
    
    Tests connectivity to all services and reports circuit breaker states
    """
    from core.startup import startup_manager
    
    pure_llm_core = startup_manager.get_pure_llm_core()
    
    if not pure_llm_core:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pure LLM Handler not available"
        )
    
    try:
        test_results = await pure_llm_core.test_circuit_breakers()
        
        return {
            **test_results,
            "message": "Circuit breaker tests completed",
            "recommendation": (
                "All services healthy" 
                if test_results['summary']['failed'] == 0 
                else f"{test_results['summary']['failed']} service(s) unavailable"
            )
        }
    except Exception as e:
        logger.error(f"Circuit breaker test failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Circuit breaker test error: {str(e)}"
        )


@router.get("/readiness")
async def readiness_probe():
    """
    Kubernetes readiness probe
    Returns 200 if ready to serve traffic
    """
    from core.startup import startup_manager
    
    if not startup_manager.initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )
    
    return {"status": "ready"}


@router.get("/liveness")
async def liveness_probe():
    """
    Kubernetes liveness probe
    Returns 200 if application is alive
    """
    return {"status": "alive"}


@router.get("/llm")
async def llm_health_check(request: Request):
    """
    UnifiedLLMService health check
    
    Returns comprehensive health status including:
    - Service availability
    - Circuit breaker state
    - Cache statistics
    - Backend status (vLLM/Groq)
    - Performance metrics
    
    This endpoint provides real-time monitoring of the centralized LLM service
    used by all handlers and API endpoints.
    """
    if not hasattr(request.app.state, 'unified_llm') or request.app.state.unified_llm is None:
        return {
            "status": "unavailable",
            "timestamp": datetime.utcnow().isoformat(),
            "error": "UnifiedLLMService not initialized",
            "message": "Service will fall back to legacy LLM clients"
        }
    
    try:
        unified_llm = request.app.state.unified_llm
        metrics = unified_llm.get_metrics()
        
        # Determine overall status
        overall_status = "healthy"
        if unified_llm.circuit_breaker_open:
            overall_status = "degraded"  # Circuit breaker open, using fallback
        
        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "service": "UnifiedLLMService",
            "backend": {
                "primary": {
                    "type": "vLLM",
                    "endpoint": unified_llm.vllm_endpoint,
                    "available": not unified_llm.circuit_breaker_open
                },
                "fallback": {
                    "type": "Groq",
                    "available": True,
                    "active": unified_llm.circuit_breaker_open
                }
            },
            "circuit_breaker": {
                "state": "open" if unified_llm.circuit_breaker_open else "closed",
                "failure_count": unified_llm.circuit_breaker_failures,
                "failure_threshold": unified_llm.circuit_breaker_threshold,
                "last_failure": getattr(unified_llm, 'last_failure_time', None)
            },
            "cache": {
                "size": len(unified_llm.cache),
                "max_size": unified_llm.cache_max_size,
                "usage_percent": (len(unified_llm.cache) / unified_llm.cache_max_size * 100) if unified_llm.cache_max_size > 0 else 0,
                "hit_rate": metrics.get("cache_hit_rate", 0),
                "hits": metrics.get("cache_hits", 0),
                "misses": metrics.get("cache_misses", 0)
            },
            "metrics": {
                "total_requests": metrics.get("total_requests", 0),
                "total_errors": metrics.get("total_errors", 0),
                "error_rate": metrics.get("error_rate", 0),
                "avg_latency_ms": metrics.get("avg_latency_ms", 0),
                "p95_latency_ms": metrics.get("p95_latency_ms", 0),
                "total_tokens": metrics.get("total_tokens", 0)
            },
            "features": {
                "caching": True,
                "circuit_breaker": True,
                "streaming": True,
                "metrics_tracking": True,
                "multi_backend": True
            }
        }
        
    except Exception as e:
        logger.error(f"UnifiedLLMService health check failed: {e}")
        return {
            "status": "error",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
            "message": "Health check failed"
        }
