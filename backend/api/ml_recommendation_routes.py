"""
API Routes for Integrated ML Recommendation System
Integrates LightGBM, NCF, and LLM for production recommendations
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging
import sys
from pathlib import Path

# Add ML directory to path
ml_dir = Path(__file__).parent.parent / 'ml'
sys.path.insert(0, str(ml_dir))

from serving.integrated_recommendation_service import IntegratedRecommendationService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/ml", tags=["ML Recommendations"])

# Global service instance (initialized on startup)
_service: Optional[IntegratedRecommendationService] = None


# Request/Response models
class RecommendationRequest(BaseModel):
    """Request for recommendations"""
    user_id: int = Field(..., description="User ID")
    query: str = Field(..., description="User query or request")
    location: Optional[Dict[str, float]] = Field(None, description="User location (lat, lon)")
    preferences: Optional[List[str]] = Field(None, description="User preferences")
    top_k: int = Field(10, ge=1, le=100, description="Number of recommendations")
    candidate_pool_size: int = Field(100, ge=10, le=500, description="Candidate pool size")


class ComponentScore(BaseModel):
    """Individual component score"""
    llm: float = Field(0.0, description="LLM semantic score")
    ncf: float = Field(0.0, description="NCF collaborative filtering score")
    lightgbm: float = Field(0.0, description="LightGBM feature-based score")


class Recommendation(BaseModel):
    """Single recommendation"""
    rank: int = Field(..., description="Rank in results")
    item_id: int = Field(..., description="Item ID")
    name: str = Field(..., description="Item name")
    category: str = Field(..., description="Item category")
    final_score: float = Field(..., description="Final ensemble score")
    component_scores: ComponentScore = Field(..., description="Component scores")
    explanation: str = Field(..., description="Human-readable explanation")
    metadata: Dict[str, Any] = Field({}, description="Additional metadata")


class RecommendationResponse(BaseModel):
    """Response with recommendations"""
    user_id: int
    query: str
    recommendations: List[Recommendation]
    total_count: int
    processing_time_ms: float
    service_info: Dict[str, Any]


class StatisticsUpdate(BaseModel):
    """Request to update user/item statistics"""
    user_stats: Optional[Dict[int, Dict[str, float]]] = None
    item_stats: Optional[Dict[int, Dict[str, float]]] = None


# Service initialization
def get_service() -> IntegratedRecommendationService:
    """
    Get or initialize the integrated recommendation service
    """
    global _service
    
    if _service is None:
        logger.info("🚀 Initializing integrated recommendation service...")
        
        # Initialize with default paths (can be configured via env vars)
        _service = IntegratedRecommendationService(
            lightgbm_model_path='models/ranker/lightgbm_ranker.pkl',
            lightgbm_feature_stats_path='models/ranker/feature_stats.pkl',
            ncf_model_path='models/ncf/best_model.pt',
            ncf_embeddings_path='models/ncf/embeddings.pkl',
            use_llm=True,
            ensemble_weights={'llm': 0.4, 'ncf': 0.3, 'lightgbm': 0.3}
        )
        
        logger.info("✅ Integrated recommendation service initialized")
    
    return _service


# API Endpoints
@router.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest,
    service: IntegratedRecommendationService = Depends(get_service)
):
    """
    Get integrated recommendations combining LLM, NCF, and LightGBM
    
    This endpoint provides the most advanced recommendations by:
    1. Using LLM for semantic understanding and candidate generation
    2. Applying NCF for collaborative filtering
    3. Re-ranking with LightGBM using rich features
    4. Combining all signals with learned ensemble weights
    
    Example:
    ```
    POST /api/ml/recommendations
    {
        "user_id": 42,
        "query": "Find me authentic Turkish breakfast places",
        "location": {"lat": 41.0082, "lon": 28.9784},
        "preferences": ["authentic", "local"],
        "top_k": 10
    }
    ```
    """
    try:
        start_time = datetime.now()
        
        # Build context
        context = {
            'location': request.location or {},
            'preferences': request.preferences or [],
            'time': datetime.now()
        }
        
        # Get recommendations
        recommendations = await service.get_recommendations(
            user_id=request.user_id,
            query=request.query,
            context=context,
            top_k=request.top_k,
            candidate_pool_size=request.candidate_pool_size
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Get service info
        service_info = service.get_service_info()
        
        # Format response
        return RecommendationResponse(
            user_id=request.user_id,
            query=request.query,
            recommendations=[
                Recommendation(
                    rank=rec['rank'],
                    item_id=rec['item_id'],
                    name=rec['name'],
                    category=rec['category'],
                    final_score=rec['final_score'],
                    component_scores=ComponentScore(**rec['component_scores']),
                    explanation=rec['explanation'],
                    metadata=rec['metadata']
                )
                for rec in recommendations
            ],
            total_count=len(recommendations),
            processing_time_ms=processing_time,
            service_info=service_info
        )
        
    except Exception as e:
        logger.error(f"❌ Error getting recommendations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/service-info")
async def get_service_info(
    service: IntegratedRecommendationService = Depends(get_service)
):
    """
    Get information about the integrated recommendation service
    
    Returns details about available components and their weights.
    """
    try:
        return service.get_service_info()
    except Exception as e:
        logger.error(f"❌ Error getting service info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update-statistics")
async def update_statistics(
    request: StatisticsUpdate,
    service: IntegratedRecommendationService = Depends(get_service)
):
    """
    Update cached user and item statistics
    
    This endpoint allows updating the LightGBM ranker's cached statistics
    for faster inference. Statistics should be precomputed from your database.
    
    Example:
    ```
    POST /api/ml/update-statistics
    {
        "user_stats": {
            "42": {
                "n_interactions": 15,
                "avg_rating": 4.3,
                "std_rating": 0.6,
                "days_since_last_interaction": 1.5
            }
        },
        "item_stats": {
            "100": {
                "popularity": 250,
                "log_popularity": 5.52,
                "avg_rating": 4.5,
                "std_rating": 0.4
            }
        }
    }
    ```
    """
    try:
        service.update_statistics(
            user_stats=request.user_stats,
            item_stats=request.item_stats
        )
        
        return {
            "status": "success",
            "message": "Statistics updated successfully",
            "user_stats_count": len(request.user_stats or {}),
            "item_stats_count": len(request.item_stats or {})
        }
        
    except Exception as e:
        logger.error(f"❌ Error updating statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ensemble-weights")
async def update_ensemble_weights(
    weights: Dict[str, float],
    service: IntegratedRecommendationService = Depends(get_service)
):
    """
    Update ensemble weights for combining model outputs
    
    Allows dynamic adjustment of how much each component contributes
    to the final recommendations.
    
    Example:
    ```
    POST /api/ml/ensemble-weights
    {
        "llm": 0.5,
        "ncf": 0.3,
        "lightgbm": 0.2
    }
    ```
    """
    try:
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight <= 0:
            raise ValueError("Total weight must be positive")
        
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
        service.ensemble_weights = normalized_weights
        
        return {
            "status": "success",
            "message": "Ensemble weights updated",
            "weights": normalized_weights
        }
        
    except Exception as e:
        logger.error(f"❌ Error updating weights: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/health")
async def health_check(
    service: IntegratedRecommendationService = Depends(get_service)
):
    """
    Health check endpoint
    
    Returns the status of all ML components.
    """
    try:
        info = service.get_service_info()
        
        return {
            "status": "healthy",
            "components": {
                "llm": "enabled" if info['llm_enabled'] else "disabled",
                "ncf": "available" if info['ncf_available'] else "unavailable",
                "lightgbm": "available" if info['lightgbm_available'] else "unavailable"
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# Backward compatibility endpoints
@router.post("/recommend")
async def recommend_legacy(
    user_id: int = Query(..., description="User ID"),
    query: str = Query(..., description="User query"),
    top_k: int = Query(10, description="Number of recommendations"),
    service: IntegratedRecommendationService = Depends(get_service)
):
    """
    Legacy recommendation endpoint (backward compatible)
    
    Simple endpoint for getting recommendations with minimal parameters.
    """
    try:
        request = RecommendationRequest(
            user_id=user_id,
            query=query,
            top_k=top_k
        )
        
        response = await get_recommendations(request, service)
        
        # Return simplified format for legacy clients
        return {
            "recommendations": [
                {
                    "id": rec.item_id,
                    "name": rec.name,
                    "category": rec.category,
                    "score": rec.final_score,
                    "explanation": rec.explanation
                }
                for rec in response.recommendations
            ]
        }
        
    except Exception as e:
        logger.error(f"❌ Error in legacy endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
