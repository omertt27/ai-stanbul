"""
Production NCF Recommendation API

FastAPI routes for ONNX-optimized NCF recommendations with:
- A/B testing
- Caching
- Monitoring
- Health checks

Author: AI Istanbul Team
Date: February 10, 2026
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging

from backend.services.onnx_ncf_service import get_onnx_ncf_service, ONNXNCFService
from backend.services.ncf_ab_testing import ABTestFramework, create_ncf_onnx_experiment

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ncf", tags=["ncf-recommendations"])


# Request/Response Models
class RecommendationRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    top_k: int = Field(5, ge=1, le=20, description="Number of recommendations")
    candidate_items: Optional[List[int]] = Field(None, description="Candidate item IDs")
    use_cache: bool = Field(True, description="Enable caching")
    ab_test_override: Optional[str] = Field(None, description="Override A/B test variant")


class RecommendationItem(BaseModel):
    id: str
    name: str
    score: float
    confidence: float
    type: str
    metadata: Dict[str, Any]


class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[RecommendationItem]
    variant: str  # A/B test variant used
    cached: bool
    latency_ms: float
    model_version: str


class MetricsResponse(BaseModel):
    service_metrics: Dict[str, Any]
    ab_test_metrics: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    healthy: bool
    onnx_model: bool
    redis_cache: bool
    errors: List[str]


# Global A/B test instance
_ab_test = None


def get_ab_test() -> ABTestFramework:
    """Get or create A/B test instance."""
    global _ab_test
    if _ab_test is None:
        service = get_onnx_ncf_service()
        _ab_test = create_ncf_onnx_experiment(
            redis_client=service.redis_client if service.enable_caching else None
        )
    return _ab_test


@router.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest,
    service: ONNXNCFService = Depends(get_onnx_ncf_service),
    ab_test: ABTestFramework = Depends(get_ab_test)
):
    """
    Get personalized recommendations for a user.
    
    **Features:**
    - ONNX-optimized inference (3-5x faster)
    - Redis caching
    - A/B testing support
    - Performance monitoring
    
    **Example:**
    ```json
    {
      "user_id": "user_123",
      "top_k": 5,
      "use_cache": true
    }
    ```
    """
    import time
    start_time = time.time()
    
    try:
        # Determine A/B test variant
        if request.ab_test_override:
            variant = request.ab_test_override
        else:
            variant = ab_test.get_variant(request.user_id)
        
        # Track impression
        ab_test.track_impression(request.user_id, variant)
        
        # Configure service for variant
        if variant == 'fallback':
            # Force fallback for this variant
            temp_predictor = service.predictor
            service.predictor = None
        
        # Get recommendations
        recommendations = service.get_recommendations(
            user_id=request.user_id,
            candidate_items=request.candidate_items,
            top_k=request.top_k,
            use_cache=request.use_cache
        )
        
        # Restore predictor if we disabled it
        if variant == 'fallback':
            service.predictor = temp_predictor
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Track latency in A/B test
        ab_test.track_latency(variant, latency_ms)
        
        # Determine if cached
        cached = request.use_cache and service.metrics['cache_hits'] > 0
        
        # Format response
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=[RecommendationItem(**rec) for rec in recommendations],
            variant=variant,
            cached=cached,
            latency_ms=round(latency_ms, 2),
            model_version="v1.0-onnx"
        )
        
    except Exception as e:
        logger.error(f"Recommendation error for user {request.user_id}: {e}")
        
        # Track error
        ab_test.track_error(variant if 'variant' in locals() else 'unknown')
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate recommendations: {str(e)}"
        )


@router.post("/track/click")
async def track_click(
    user_id: str,
    item_id: str,
    variant: Optional[str] = None,
    ab_test: ABTestFramework = Depends(get_ab_test)
):
    """
    Track when user clicks a recommendation.
    
    **Args:**
    - user_id: User identifier
    - item_id: Clicked item ID
    - variant: A/B test variant (auto-detected if None)
    """
    try:
        ab_test.track_click(user_id, variant, item_id)
        return {"status": "success", "message": "Click tracked"}
    except Exception as e:
        logger.error(f"Click tracking error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/track/feedback")
async def track_feedback(
    user_id: str,
    feedback_type: str = Query(..., regex="^(positive|negative)$"),
    variant: Optional[str] = None,
    ab_test: ABTestFramework = Depends(get_ab_test)
):
    """
    Track user feedback on recommendations.
    
    **Args:**
    - user_id: User identifier
    - feedback_type: 'positive' or 'negative'
    - variant: A/B test variant (auto-detected if None)
    """
    try:
        ab_test.track_feedback(user_id, feedback_type, variant)
        return {"status": "success", "message": "Feedback tracked"}
    except Exception as e:
        logger.error(f"Feedback tracking error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    service: ONNXNCFService = Depends(get_onnx_ncf_service),
    ab_test: ABTestFramework = Depends(get_ab_test)
):
    """
    Get service and A/B test metrics.
    
    **Returns:**
    - Service performance metrics
    - A/B test results
    - Cache statistics
    """
    return MetricsResponse(
        service_metrics=service.get_metrics(),
        ab_test_metrics=ab_test.get_metrics()
    )


@router.get("/ab-test/report")
async def get_ab_test_report(ab_test: ABTestFramework = Depends(get_ab_test)):
    """
    Get detailed A/B test report.
    
    **Returns:**
    - Formatted text report with:
      - Variant distribution
      - Metrics per variant
      - Statistical significance
    """
    report = ab_test.get_report()
    return {"report": report, "raw_metrics": ab_test.get_metrics()}


@router.post("/cache/invalidate")
async def invalidate_cache(
    user_id: Optional[str] = None,
    service: ONNXNCFService = Depends(get_onnx_ncf_service)
):
    """
    Invalidate recommendation cache.
    
    **Args:**
    - user_id: User to invalidate (None = all users)
    """
    try:
        service.invalidate_cache(user_id)
        message = f"Cache invalidated for user {user_id}" if user_id else "All cache invalidated"
        return {"status": "success", "message": message}
    except Exception as e:
        logger.error(f"Cache invalidation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check(service: ONNXNCFService = Depends(get_onnx_ncf_service)):
    """
    Health check endpoint.
    
    **Returns:**
    - Overall health status
    - ONNX model status
    - Redis cache status
    - Any errors
    """
    health = service.health_check()
    
    return HealthResponse(
        status="healthy" if health['healthy'] else "degraded",
        healthy=health['healthy'],
        onnx_model=health['onnx_model'],
        redis_cache=health['redis_cache'],
        errors=health['errors']
    )


@router.post("/metrics/reset")
async def reset_metrics(
    service: ONNXNCFService = Depends(get_onnx_ncf_service)
):
    """
    Reset service metrics.
    
    **Use with caution** - clears all performance metrics.
    """
    try:
        service.reset_metrics()
        return {"status": "success", "message": "Metrics reset"}
    except Exception as e:
        logger.error(f"Metrics reset error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Include router in main app:
# from backend.api.ncf_production import router as ncf_router
# app.include_router(ncf_router)
