"""
Production Recommendations API

FastAPI router for production NCF recommendations with A/B testing,
personalization, and performance monitoring.

Endpoints:
- GET /api/recommendations/personalized: Personalized NCF recommendations
- GET /api/recommendations/similar: Similar items (content-based)
- GET /api/recommendations/trending: Trending items with NCF boost
- POST /api/recommendations/feedback: User feedback collection
- GET /api/recommendations/health: Service health check

Author: AI Istanbul Team
Date: February 12, 2026
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import time

from services.production_ncf_service import get_ncf_service
from services.ab_testing_service import get_ab_testing_service
from database import get_db
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/recommendations", tags=["recommendations"])


# ==================== Request/Response Models ====================

class RecommendationRequest(BaseModel):
    """Request for personalized recommendations."""
    user_id: int = Field(..., description="User ID for personalization")
    limit: int = Field(10, ge=1, le=50, description="Number of recommendations")
    category: Optional[str] = Field(None, description="Filter by category")
    exclude_seen: bool = Field(True, description="Exclude already seen items")
    include_scores: bool = Field(False, description="Include prediction scores")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context (location, time, etc.)")


class RecommendationItem(BaseModel):
    """Single recommendation item."""
    item_id: int
    item_name: str
    category: str
    score: Optional[float] = None
    reason: Optional[str] = None
    ab_variant: Optional[str] = None


class RecommendationResponse(BaseModel):
    """Response with recommendations."""
    recommendations: List[RecommendationItem]
    user_id: int
    total: int
    algorithm: str
    ab_test_id: Optional[str] = None
    ab_variant: Optional[str] = None
    cached: bool
    latency_ms: float
    timestamp: datetime


class FeedbackRequest(BaseModel):
    """User feedback on recommendations."""
    user_id: int
    item_id: int
    feedback_type: str = Field(..., description="click, like, dislike, purchase, skip")
    ab_test_id: Optional[str] = None
    ab_variant: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Service health status."""
    status: str
    model_loaded: bool
    cache_available: bool
    ab_testing_enabled: bool
    stats: Dict[str, Any]


# ==================== API Endpoints ====================

@router.get("/personalized", response_model=RecommendationResponse)
async def get_personalized_recommendations(
    user_id: int = Query(..., description="User ID"),
    limit: int = Query(10, ge=1, le=50),
    category: Optional[str] = Query(None),
    exclude_seen: bool = Query(True),
    include_scores: bool = Query(False),
    db: Session = Depends(get_db)
):
    """
    Get personalized NCF recommendations with A/B testing.
    
    This endpoint:
    1. Assigns user to A/B test variant (if active)
    2. Gets NCF predictions or fallback
    3. Tracks performance metrics
    4. Returns recommendations with metadata
    """
    start_time = time.time()
    
    try:
        # Get services
        ncf_service = get_ncf_service()
        ab_service = get_ab_testing_service()
        
        # A/B test assignment
        ab_test_id = None
        ab_variant = None
        
        if ab_service.has_active_tests():
            assignment = ab_service.assign_user(user_id, "ncf_recommendations")
            ab_test_id = assignment.get("test_id")
            ab_variant = assignment.get("variant")
            
            logger.info(f"User {user_id} assigned to A/B test: {ab_test_id}, variant: {ab_variant}")
        
        # Get recommendations (variant affects algorithm params)
        if ab_variant == "control":
            # Control: Use popular items (no personalization)
            recommendations = await _get_popular_items(db, limit, category)
            algorithm = "popular"
            cached = False
        else:
            # Treatment: Use NCF personalization
            recommendations = await ncf_service.get_recommendations(
                user_id=user_id,
                top_k=limit,
                exclude_interacted=exclude_seen
            )
            algorithm = "ncf"
            # Note: NCF service handles caching internally but doesn't expose cached status
            # TODO: Enhance service to return metadata including cache status
            cached = False
        
        # Filter by category if requested
        if category:
            recommendations = [r for r in recommendations if r.get("category") == category]
        
        # Format response
        items = []
        for rec in recommendations[:limit]:
            item = RecommendationItem(
                item_id=rec["item_id"],
                item_name=rec.get("name", f"Item {rec['item_id']}"),
                category=rec.get("category", "unknown"),
                score=rec.get("score") if include_scores else None,
                reason=rec.get("reason"),
                ab_variant=ab_variant
            )
            items.append(item)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Track A/B test metrics
        if ab_test_id:
            ab_service.track_metric(
                test_id=ab_test_id,
                variant=ab_variant,
                metric_name="api_latency_ms",
                value=latency_ms
            )
        
        return RecommendationResponse(
            recommendations=items,
            user_id=user_id,
            total=len(items),
            algorithm=algorithm,
            ab_test_id=ab_test_id,
            ab_variant=ab_variant,
            cached=cached,
            latency_ms=latency_ms,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Error getting recommendations for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback")
async def submit_feedback(
    feedback: FeedbackRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Submit user feedback on recommendations.
    
    Feedback is used for:
    - A/B test conversion tracking
    - Model retraining data
    - Real-time performance monitoring
    """
    try:
        # Track A/B test conversion
        if feedback.ab_test_id and feedback.ab_variant:
            ab_service = get_ab_testing_service()
            
            # Map feedback type to conversion
            is_conversion = feedback.feedback_type in ["click", "like", "purchase"]
            
            if is_conversion:
                ab_service.track_conversion(
                    test_id=feedback.ab_test_id,
                    variant=feedback.ab_variant
                )
        
        # Store feedback for retraining (background task)
        background_tasks.add_task(
            _store_feedback_for_training,
            feedback.user_id,
            feedback.item_id,
            feedback.feedback_type,
            feedback.context
        )
        
        return {
            "status": "success",
            "message": "Feedback recorded",
            "user_id": feedback.user_id,
            "item_id": feedback.item_id
        }
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check production NCF service health.
    
    Returns:
    - Model status
    - Cache availability
    - A/B testing status
    - Performance stats
    """
    try:
        ncf_service = get_ncf_service()
        ab_service = get_ab_testing_service()
        
        # Get service stats
        ncf_stats = ncf_service.get_stats()
        ab_stats = ab_service.get_stats() if ab_service else {}
        
        return HealthResponse(
            status="healthy",
            model_loaded=ncf_stats.get("model_loaded", False),
            cache_available=ncf_stats.get("cache_available", False),
            ab_testing_enabled=len(ab_stats.get("active_tests", [])) > 0,
            stats={
                "ncf": ncf_stats,
                "ab_testing": ab_stats
            }
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            cache_available=False,
            ab_testing_enabled=False,
            stats={"error": str(e)}
        )


@router.get("/trending", response_model=RecommendationResponse)
async def get_trending_recommendations(
    user_id: Optional[int] = Query(None),
    limit: int = Query(10, ge=1, le=50),
    hours: int = Query(24, ge=1, le=168),
    db: Session = Depends(get_db)
):
    """
    Get trending items with optional NCF personalization boost.
    
    If user_id provided: trending items re-ranked by NCF scores
    If no user_id: pure trending based on recent interactions
    """
    start_time = time.time()
    
    try:
        # Get trending items from database
        trending_items = await _get_trending_items(db, hours, limit * 2)
        
        # If user_id provided, re-rank with NCF
        if user_id:
            ncf_service = get_ncf_service()
            
            # Get NCF scores for trending items
            scored_items = []
            for item in trending_items:
                score = ncf_service.predict_single(user_id, item["item_id"])
                scored_items.append({**item, "score": score})
            
            # Sort by NCF score
            scored_items.sort(key=lambda x: x["score"], reverse=True)
            recommendations = scored_items[:limit]
            algorithm = "trending_ncf"
        else:
            recommendations = trending_items[:limit]
            algorithm = "trending"
        
        # Format response
        items = [
            RecommendationItem(
                item_id=rec["item_id"],
                item_name=rec.get("name", f"Item {rec['item_id']}"),
                category=rec.get("category", "unknown"),
                score=rec.get("score"),
                reason="Trending now"
            )
            for rec in recommendations
        ]
        
        latency_ms = (time.time() - start_time) * 1000
        
        return RecommendationResponse(
            recommendations=items,
            user_id=user_id or 0,
            total=len(items),
            algorithm=algorithm,
            ab_test_id=None,
            ab_variant=None,
            cached=False,
            latency_ms=latency_ms,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Error getting trending items: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Helper Functions ====================

async def _get_popular_items(db: Session, limit: int, category: Optional[str] = None) -> List[Dict]:
    """Get popular items (fallback for control group)."""
    # TODO: Implement based on your database schema
    # This is a placeholder
    return [
        {
            "item_id": i,
            "name": f"Popular Item {i}",
            "category": category or "restaurant",
            "score": 1.0 - (i * 0.01)
        }
        for i in range(1, limit + 1)
    ]


async def _get_trending_items(db: Session, hours: int, limit: int) -> List[Dict]:
    """Get trending items from recent interactions."""
    # TODO: Implement based on your database schema
    # This is a placeholder
    return [
        {
            "item_id": i,
            "name": f"Trending Item {i}",
            "category": "restaurant",
            "trend_score": 1.0 - (i * 0.01)
        }
        for i in range(1, limit + 1)
    ]


async def _store_feedback_for_training(
    user_id: int,
    item_id: int,
    feedback_type: str,
    context: Optional[Dict]
):
    """Store feedback for model retraining (background task)."""
    try:
        from services.training_data_collector import get_training_data_collector
        
        collector = get_training_data_collector()
        collector.store_interaction(
            user_id=user_id,
            item_id=item_id,
            interaction_type=feedback_type,
            context=context
        )
        
        logger.info(f"Stored feedback for training: user={user_id}, item={item_id}, type={feedback_type}")
        
    except Exception as e:
        logger.error(f"Error storing feedback for training: {e}")


# ==================== Module Initialization ====================

import time

logger.info("✅ Production Recommendations API loaded")
