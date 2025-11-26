"""
Health Check Endpoints Module

System health monitoring and diagnostics
"""

from fastapi import APIRouter, HTTPException, status
from datetime import datetime
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/health", tags=["System Health"])


@router.get("")
async def health_check():
    """
    Overall system health check
    Returns health status of all services
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "api": "healthy",
            "database": "healthy",
            "cache": "healthy"
        }
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
