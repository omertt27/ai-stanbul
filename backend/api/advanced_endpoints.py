"""
FastAPI endpoints for advanced caching, A/B testing, and user feedback
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
import redis.asyncio as redis
import logging

from ..services.caching.advanced_cache import AdvancedCacheSystem
from ..services.ab_testing.experiment_manager import ABTestingManager, Experiment
from ..services.feedback.feedback_collector import FeedbackCollector, FeedbackType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["advanced"])


# Pydantic Models
class FeedbackRequest(BaseModel):
    session_id: str
    query: str
    response: str
    feedback_type: str = Field(..., description="thumbs_up, thumbs_down, star_rating, comment")
    rating: Optional[int] = Field(None, ge=1, le=5)
    comment: Optional[str] = None
    language: str = "en"
    metadata: Optional[Dict[str, Any]] = None


class ExperimentCreate(BaseModel):
    name: str
    description: str
    variants: List[Dict[str, Any]]
    traffic_split: Optional[List[float]] = None


class CacheWarmRequest(BaseModel):
    queries: List[Dict[str, str]]


# Dependency: Get Redis client
async def get_redis() -> redis.Redis:
    """Get Redis connection"""
    return await redis.from_url(
        "redis://localhost:6379",
        encoding="utf-8",
        decode_responses=True
    )


# Feedback Endpoints
@router.post("/feedback", status_code=201)
async def submit_feedback(
    feedback: FeedbackRequest,
    redis_client: redis.Redis = Depends(get_redis)
):
    """
    Submit user feedback
    
    Example:
    ```json
    {
        "session_id": "sess_12345",
        "query": "Best restaurants in Taksim?",
        "response": "Here are some great restaurants...",
        "feedback_type": "thumbs_up",
        "language": "en",
        "metadata": {
            "model": "llama-3.1-8b",
            "response_time": 2.5
        }
    }
    ```
    """
    try:
        collector = FeedbackCollector(redis_client)
        
        feedback_type = FeedbackType(feedback.feedback_type)
        
        feedback_id = collector.submit_feedback(
            session_id=feedback.session_id,
            query=feedback.query,
            response=feedback.response,
            feedback_type=feedback_type,
            rating=feedback.rating,
            comment=feedback.comment,
            language=feedback.language,
            metadata=feedback.metadata
        )
        
        return {
            "status": "success",
            "feedback_id": feedback_id,
            "message": "Thank you for your feedback!"
        }
    
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feedback/analytics")
async def get_feedback_analytics(
    days: int = 7,
    redis_client: redis.Redis = Depends(get_redis)
):
    """
    Get feedback analytics for the last N days
    
    Query params:
    - days: Number of days to analyze (default: 7)
    """
    try:
        collector = FeedbackCollector(redis_client)
        
        start_date = datetime.now() - timedelta(days=days)
        analytics = collector.get_analytics(start_date=start_date)
        
        return {
            "status": "success",
            "analytics": analytics
        }
    
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feedback/negative")
async def get_negative_feedback(
    limit: int = 50,
    redis_client: redis.Redis = Depends(get_redis)
):
    """
    Get negative feedback for improvement analysis
    
    Query params:
    - limit: Maximum number of entries (default: 50)
    """
    try:
        collector = FeedbackCollector(redis_client)
        negative = collector.get_negative_feedback_for_improvement(limit=limit)
        
        return {
            "status": "success",
            "count": len(negative),
            "feedback": negative
        }
    
    except Exception as e:
        logger.error(f"Error getting negative feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# A/B Testing Endpoints
@router.post("/experiments", status_code=201)
async def create_experiment(
    experiment: ExperimentCreate,
    redis_client: redis.Redis = Depends(get_redis)
):
    """
    Create a new A/B testing experiment
    
    Example:
    ```json
    {
        "name": "Temperature Test",
        "description": "Testing different temperature settings",
        "variants": [
            {
                "name": "control",
                "temperature": 0.7,
                "max_tokens": 512
            },
            {
                "name": "creative",
                "temperature": 0.9,
                "max_tokens": 512
            }
        ],
        "traffic_split": [0.5, 0.5]
    }
    ```
    """
    try:
        manager = ABTestingManager(redis_client)
        
        exp = manager.create_experiment(
            name=experiment.name,
            description=experiment.description,
            variants=experiment.variants,
            traffic_split=experiment.traffic_split
        )
        
        return {
            "status": "success",
            "experiment": exp.to_dict()
        }
    
    except Exception as e:
        logger.error(f"Error creating experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments")
async def list_experiments(
    active_only: bool = False,
    redis_client: redis.Redis = Depends(get_redis)
):
    """
    List all experiments
    
    Query params:
    - active_only: Only return active experiments (default: false)
    """
    try:
        manager = ABTestingManager(redis_client)
        experiments = manager.list_experiments(active_only=active_only)
        
        return {
            "status": "success",
            "count": len(experiments),
            "experiments": [exp.to_dict() for exp in experiments]
        }
    
    except Exception as e:
        logger.error(f"Error listing experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/{experiment_id}")
async def get_experiment(
    experiment_id: str,
    redis_client: redis.Redis = Depends(get_redis)
):
    """Get experiment by ID"""
    try:
        manager = ABTestingManager(redis_client)
        experiment = manager.get_experiment(experiment_id)
        
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        return {
            "status": "success",
            "experiment": experiment.to_dict()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/{experiment_id}/results")
async def get_experiment_results(
    experiment_id: str,
    redis_client: redis.Redis = Depends(get_redis)
):
    """Get experiment results and statistics"""
    try:
        manager = ABTestingManager(redis_client)
        results = manager.get_results(experiment_id)
        
        if not results:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        return {
            "status": "success",
            "results": results
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/experiments/{experiment_id}/stop")
async def stop_experiment(
    experiment_id: str,
    redis_client: redis.Redis = Depends(get_redis)
):
    """Stop an experiment"""
    try:
        manager = ABTestingManager(redis_client)
        manager.stop_experiment(experiment_id)
        
        return {
            "status": "success",
            "message": "Experiment stopped"
        }
    
    except Exception as e:
        logger.error(f"Error stopping experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/{experiment_id}/assign/{user_id}")
async def assign_variant(
    experiment_id: str,
    user_id: str,
    redis_client: redis.Redis = Depends(get_redis)
):
    """Assign user to a variant"""
    try:
        manager = ABTestingManager(redis_client)
        variant = manager.assign_variant(experiment_id, user_id)
        
        if not variant:
            raise HTTPException(
                status_code=404,
                detail="Experiment not found or not active"
            )
        
        return {
            "status": "success",
            "variant": variant
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error assigning variant: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Cache Endpoints
@router.get("/cache/stats")
async def get_cache_stats(redis_client: redis.Redis = Depends(get_redis)):
    """Get cache performance statistics"""
    try:
        cache = AdvancedCacheSystem(redis_client)
        stats = cache.get_stats()
        
        return {
            "status": "success",
            "stats": stats
        }
    
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/warm")
async def warm_cache(
    request: CacheWarmRequest,
    background_tasks: BackgroundTasks,
    redis_client: redis.Redis = Depends(get_redis)
):
    """
    Warm cache with common queries
    
    Example:
    ```json
    {
        "queries": [
            {"query": "Best restaurants in Taksim", "language": "en"},
            {"query": "How to get to Hagia Sophia", "language": "en"}
        ]
    }
    ```
    """
    try:
        cache = AdvancedCacheSystem(redis_client)
        
        # Run cache warming in background
        background_tasks.add_task(cache.warm_cache, request.queries)
        
        return {
            "status": "success",
            "message": f"Warming cache with {len(request.queries)} queries"
        }
    
    except Exception as e:
        logger.error(f"Error warming cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/invalidate")
async def invalidate_cache(
    pattern: Optional[str] = None,
    redis_client: redis.Redis = Depends(get_redis)
):
    """
    Invalidate cache entries
    
    Query params:
    - pattern: Redis key pattern (default: all cache keys)
    """
    try:
        cache = AdvancedCacheSystem(redis_client)
        count = cache.invalidate(pattern=pattern)
        
        return {
            "status": "success",
            "message": f"Invalidated {count} cache entries"
        }
    
    except Exception as e:
        logger.error(f"Error invalidating cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoint
@router.get("/health")
async def health_check(redis_client: redis.Redis = Depends(get_redis)):
    """Health check for advanced features"""
    try:
        # Check Redis connection
        await redis_client.ping()
        
        return {
            "status": "healthy",
            "services": {
                "redis": "connected",
                "caching": "operational",
                "ab_testing": "operational",
                "feedback": "operational"
            }
        }
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }
