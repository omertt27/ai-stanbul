"""
Monitoring API Endpoints

Provides health checks, metrics, and system status for the NCF system.

Author: AI Istanbul Team
Date: February 10, 2026
"""

import logging
import time
import psutil
from typing import Dict, Any, List
from fastapi import APIRouter, Response, HTTPException
from pydantic import BaseModel

from backend.services.monitoring.metrics_collector import get_metrics_collector
from backend.services.production_ncf_service import get_production_ncf_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])


# === Response Models ===

class HealthStatus(BaseModel):
    """Health check response."""
    status: str
    timestamp: float
    version: str
    components: Dict[str, str]


class MetricsSummary(BaseModel):
    """Metrics summary response."""
    total_requests: int
    avg_latency_ms: float
    cache_hit_rate: float
    model_status: Dict[str, bool]
    error_rate: float


class SystemStatus(BaseModel):
    """System resource status."""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    uptime_seconds: float


# === Endpoints ===

@router.get("/health", response_model=HealthStatus)
async def health_check() -> HealthStatus:
    """
    Health check endpoint.
    
    Returns system health status including all components.
    """
    metrics = get_metrics_collector()
    ncf_service = get_production_ncf_service()
    
    # Check component health
    components = {
        "api": "healthy",
        "metrics": "healthy",
        "ncf_service": "healthy" if ncf_service else "unavailable"
    }
    
    # Check if models are loaded
    try:
        if ncf_service and hasattr(ncf_service, 'onnx_predictor'):
            components["onnx_model"] = "healthy"
        else:
            components["onnx_model"] = "unavailable"
    except Exception as e:
        logger.error(f"Model health check failed: {e}")
        components["onnx_model"] = "unhealthy"
    
    # Overall status
    status = "healthy" if all(
        s in ["healthy", "degraded"] for s in components.values()
    ) else "unhealthy"
    
    return HealthStatus(
        status=status,
        timestamp=time.time(),
        version="2.0.0-phase2",
        components=components
    )


@router.get("/metrics")
async def prometheus_metrics():
    """
    Prometheus metrics endpoint.
    
    Returns metrics in Prometheus text format.
    """
    metrics = get_metrics_collector()
    
    return Response(
        content=metrics.get_metrics(),
        media_type=metrics.get_content_type()
    )


@router.get("/metrics/summary", response_model=Dict[str, Any])
async def metrics_summary() -> Dict[str, Any]:
    """
    Get human-readable metrics summary.
    
    Returns key metrics in JSON format.
    """
    metrics = get_metrics_collector()
    
    return {
        "status": "active",
        "metrics_available": True,
        "collector_info": metrics.get_summary(),
        "note": "Use /api/monitoring/metrics for Prometheus format"
    }


@router.get("/system", response_model=SystemStatus)
async def system_status() -> SystemStatus:
    """
    Get system resource usage.
    
    Returns CPU, memory, disk usage and uptime.
    """
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Calculate uptime (boot time to now)
        boot_time = psutil.boot_time()
        uptime_seconds = time.time() - boot_time
        
        return SystemStatus(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_percent=disk.percent,
            uptime_seconds=uptime_seconds
        )
    except Exception as e:
        logger.error(f"System status check failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system status")


@router.get("/model/status")
async def model_status() -> Dict[str, Any]:
    """
    Get detailed model status.
    
    Returns information about loaded models and their status.
    """
    ncf_service = get_production_ncf_service()
    
    if not ncf_service:
        return {
            "status": "unavailable",
            "message": "NCF service not initialized"
        }
    
    status = {
        "service_initialized": True,
        "onnx_enabled": ncf_service.use_onnx,
        "cache_enabled": ncf_service.cache_enabled,
        "models": {}
    }
    
    # Check ONNX model
    if hasattr(ncf_service, 'onnx_predictor') and ncf_service.onnx_predictor:
        try:
            model_info = ncf_service.onnx_predictor.get_model_info()
            status["models"]["onnx"] = {
                "loaded": True,
                "num_users": model_info.get('num_users'),
                "num_items": model_info.get('num_items'),
                "embedding_dim": model_info.get('embedding_dim'),
                "model_size_mb": model_info.get('onnx_info', {}).get('model_size_bytes', 0) / (1024 * 1024)
            }
        except Exception as e:
            status["models"]["onnx"] = {
                "loaded": False,
                "error": str(e)
            }
    else:
        status["models"]["onnx"] = {"loaded": False}
    
    # Check PyTorch model (fallback)
    if hasattr(ncf_service, 'ncf_model') and ncf_service.ncf_model:
        status["models"]["pytorch"] = {
            "loaded": True,
            "available_as_fallback": True
        }
    else:
        status["models"]["pytorch"] = {"loaded": False}
    
    return status


@router.get("/cache/stats")
async def cache_stats() -> Dict[str, Any]:
    """
    Get cache statistics.
    
    Returns cache hit rate, size, and performance metrics.
    """
    ncf_service = get_production_ncf_service()
    
    if not ncf_service or not ncf_service.cache_enabled:
        return {
            "enabled": False,
            "message": "Caching is disabled"
        }
    
    try:
        stats = ncf_service.get_stats()
        cache_stats = stats.get('cache', {})
        
        total_requests = cache_stats.get('hits', 0) + cache_stats.get('misses', 0)
        hit_rate = (cache_stats.get('hits', 0) / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "enabled": True,
            "hits": cache_stats.get('hits', 0),
            "misses": cache_stats.get('misses', 0),
            "hit_rate_percent": round(hit_rate, 2),
            "total_requests": total_requests,
            "ttl_seconds": ncf_service.cache_ttl if hasattr(ncf_service, 'cache_ttl') else 3600
        }
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        return {
            "enabled": True,
            "error": str(e)
        }


@router.get("/performance/latency")
async def latency_stats() -> Dict[str, Any]:
    """
    Get latency statistics.
    
    Returns p50, p95, p99 latency for different operations.
    """
    ncf_service = get_production_ncf_service()
    
    if not ncf_service:
        return {
            "status": "unavailable",
            "message": "NCF service not initialized"
        }
    
    stats = ncf_service.get_stats()
    
    return {
        "inference": {
            "avg_ms": stats.get('avg_inference_time_ms', 0),
            "note": "Average inference time per request"
        },
        "total_request": {
            "avg_ms": stats.get('avg_latency_ms', 0),
            "note": "Total request latency including cache lookup"
        },
        "recommendations": {
            "total_requests": stats.get('total_requests', 0),
            "onnx_requests": stats.get('onnx_requests', 0),
            "fallback_requests": stats.get('fallback_requests', 0)
        }
    }


@router.post("/alert/test")
async def test_alert() -> Dict[str, str]:
    """
    Test alert system.
    
    Triggers a test alert to verify monitoring is working.
    """
    metrics = get_metrics_collector()
    
    # Record a test error
    metrics.record_error("TestError", "monitoring_api")
    
    logger.warning("⚠️ Test alert triggered via API")
    
    return {
        "status": "success",
        "message": "Test alert triggered. Check logs and metrics."
    }


@router.get("/dashboard/data")
async def dashboard_data() -> Dict[str, Any]:
    """
    Get comprehensive dashboard data.
    
    Returns all metrics needed for a monitoring dashboard.
    """
    ncf_service = get_production_ncf_service()
    metrics = get_metrics_collector()
    
    # Get system stats
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    
    # Get service stats
    service_stats = ncf_service.get_stats() if ncf_service else {}
    
    return {
        "timestamp": time.time(),
        "system": {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_gb": memory.used / (1024**3),
            "memory_total_gb": memory.total / (1024**3)
        },
        "service": {
            "total_requests": service_stats.get('total_requests', 0),
            "onnx_requests": service_stats.get('onnx_requests', 0),
            "fallback_requests": service_stats.get('fallback_requests', 0),
            "avg_latency_ms": service_stats.get('avg_latency_ms', 0)
        },
        "cache": {
            "enabled": ncf_service.cache_enabled if ncf_service else False,
            "hits": service_stats.get('cache', {}).get('hits', 0),
            "misses": service_stats.get('cache', {}).get('misses', 0)
        },
        "models": {
            "onnx_loaded": hasattr(ncf_service, 'onnx_predictor') and ncf_service.onnx_predictor is not None if ncf_service else False,
            "pytorch_available": hasattr(ncf_service, 'ncf_model') and ncf_service.ncf_model is not None if ncf_service else False
        }
    }
