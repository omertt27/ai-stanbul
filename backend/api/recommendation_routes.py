"""
Recommendation API Routes with A/B Testing Integration
Week 3-4: Production-ready recommendation serving with experiments
"""

from fastapi import APIRouter, HTTPException, Header, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import time
import logging

from backend.services.hidden_gems_handler import get_hidden_gems_handler
from backend.services.recommendation_ab_testing import get_ab_test_manager
from backend.services.realtime_feedback_loop import get_realtime_feedback_loop

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/recommendations", tags=["Recommendations"])

# ═══════════════════════════════════════════════════════════════════
# REQUEST/RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════

class RecommendationRequest(BaseModel):
    """Request for personalized recommendations"""
    user_id: str = Field(..., description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    location: Optional[str] = Field(None, description="Location filter (neighborhood)")
    gem_type: Optional[str] = Field(None, description="Type filter (cafe, nature, etc)")
    limit: int = Field(10, ge=1, le=50, description="Number of recommendations")
    enable_ab_test: bool = Field(True, description="Enable A/B testing")

class RecommendationItem(BaseModel):
    """Single recommendation item"""
    id: str
    name: str
    type: str
    description: str
    score: float
    metadata: Dict

class RecommendationResponse(BaseModel):
    """Response with personalized recommendations"""
    user_id: str
    session_id: str
    items: List[RecommendationItem]
    ab_test_variant: Optional[str] = None
    ab_test_experiment: Optional[str] = None
    response_time: float
    personalized: bool
    method: str

class InteractionRequest(BaseModel):
    """User interaction with a recommendation"""
    user_id: str
    session_id: str
    item_id: str
    interaction_type: str = Field(..., description="view, click, like, share, save")
    ab_test_variant: Optional[str] = None
    timestamp: Optional[float] = None

class InteractionResponse(BaseModel):
    """Confirmation of interaction tracking"""
    success: bool
    message: str
    learning_updated: bool

# ═══════════════════════════════════════════════════════════════════
# RECOMMENDATION ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.post("/personalized", response_model=RecommendationResponse)
async def get_personalized_recommendations(
    request: RecommendationRequest,
    user_agent: Optional[str] = Header(None)
):
    """
    Get personalized hidden gem recommendations with optional A/B testing
    
    This endpoint:
    1. Gets user into an A/B test variant (if enabled)
    2. Retrieves personalized recommendations using online learning
    3. Applies variant-specific ranking/filtering
    4. Returns recommendations with A/B test metadata
    """
    start_time = time.time()
    
    try:
        # Initialize services
        handler = get_hidden_gems_handler()
        ab_manager = get_ab_test_manager()
        
        # Generate session ID if not provided
        session_id = request.session_id or f"session_{int(time.time())}_{request.user_id}"
        
        # A/B Test Assignment (if enabled)
        ab_variant = None
        ab_experiment = None
        
        if request.enable_ab_test:
            try:
                assignment = ab_manager.assign_user(
                    user_id=request.user_id,
                    experiment_name="recommendation_algorithm_v1"
                )
                ab_variant = assignment['variant']
                ab_experiment = assignment['experiment']
                
                logger.info(f"👥 User {request.user_id} assigned to variant: {ab_variant}")
            except Exception as e:
                logger.warning(f"A/B test assignment failed: {e}, continuing without A/B test")
        
        # Get personalized recommendations
        # This uses the online learning feedback loop internally
        raw_recommendations = handler.get_personalized_recommendations(
            user_id=request.user_id,
            location=request.location,
            gem_type=request.gem_type,
            limit=request.limit * 2,  # Get more for A/B variant filtering
            session_id=session_id
        )
        
        # Apply A/B test variant logic
        if ab_variant and ab_variant != "control":
            raw_recommendations = _apply_variant_logic(
                raw_recommendations,
                ab_variant,
                request.limit
            )
        else:
            raw_recommendations = raw_recommendations[:request.limit]
        
        # Format response
        items = [
            RecommendationItem(
                id=rec.get('id', rec.get('name', '')),
                name=rec.get('name', 'Unknown'),
                type=rec.get('type', 'general'),
                description=rec.get('description', ''),
                score=rec.get('_personalization_score', rec.get('_relevance_score', 0.0)),
                metadata={
                    'location': rec.get('neighborhood', rec.get('location', 'Unknown')),
                    'best_time': rec.get('best_time', ''),
                    'cost': rec.get('cost', ''),
                    'hidden_factor': rec.get('hidden_factor', 0.0)
                }
            )
            for rec in raw_recommendations
        ]
        
        response_time = time.time() - start_time
        
        logger.info(
            f"✅ Served {len(items)} recommendations to user {request.user_id} "
            f"(variant: {ab_variant}, time: {response_time:.3f}s)"
        )
        
        return RecommendationResponse(
            user_id=request.user_id,
            session_id=session_id,
            items=items,
            ab_test_variant=ab_variant,
            ab_test_experiment=ab_experiment,
            response_time=response_time,
            personalized=True,
            method="online_learning_with_ab_test" if ab_variant else "online_learning"
        )
        
    except Exception as e:
        logger.error(f"❌ Recommendation error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate recommendations: {str(e)}"
        )


@router.post("/interaction", response_model=InteractionResponse)
async def track_interaction(request: InteractionRequest):
    """
    Track user interaction with a recommendation
    
    This endpoint:
    1. Records the interaction in the feedback loop
    2. Updates A/B test metrics
    3. Triggers online learning model update
    """
    try:
        feedback_loop = get_realtime_feedback_loop()
        ab_manager = get_ab_test_manager()
        
        timestamp = request.timestamp or time.time()
        
        # Convert interaction type to reward
        reward = _interaction_to_reward(request.interaction_type)
        
        # Record in feedback loop (triggers online learning update)
        feedback_loop.record_feedback(
            user_id=request.user_id,
            item_id=request.item_id,
            reward=reward,
            timestamp=timestamp
        )
        
        # Record A/B test outcome
        if request.ab_test_variant:
            try:
                ab_manager.record_outcome(
                    user_id=request.user_id,
                    experiment_name="recommendation_algorithm_v1",
                    metric_name=f"interaction_{request.interaction_type}",
                    value=reward
                )
            except Exception as e:
                logger.warning(f"Failed to record A/B outcome: {e}")
        
        logger.info(
            f"📊 Recorded {request.interaction_type} interaction: "
            f"user={request.user_id}, item={request.item_id}, reward={reward}"
        )
        
        return InteractionResponse(
            success=True,
            message=f"Interaction recorded: {request.interaction_type}",
            learning_updated=True
        )
        
    except Exception as e:
        logger.error(f"❌ Interaction tracking error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to track interaction: {str(e)}"
        )


@router.get("/popular", response_model=RecommendationResponse)
async def get_popular_recommendations(
    location: Optional[str] = Query(None, description="Location filter"),
    gem_type: Optional[str] = Query(None, description="Type filter"),
    limit: int = Query(10, ge=1, le=50),
    user_id: Optional[str] = Query(None, description="User ID for tracking")
):
    """
    Get popular (non-personalized) recommendations
    
    Useful for new users or as a fallback when personalization isn't available
    """
    start_time = time.time()
    
    try:
        handler = get_hidden_gems_handler()
        
        # Get static recommendations (no personalization)
        raw_recommendations = handler.get_hidden_gems(
            location=location,
            gem_type=gem_type,
            limit=limit
        )
        
        # Format response
        items = [
            RecommendationItem(
                id=rec.get('id', rec.get('name', '')),
                name=rec.get('name', 'Unknown'),
                type=rec.get('type', 'general'),
                description=rec.get('description', ''),
                score=rec.get('_relevance_score', 0.0),
                metadata={
                    'location': rec.get('neighborhood', rec.get('location', 'Unknown')),
                    'best_time': rec.get('best_time', ''),
                    'cost': rec.get('cost', ''),
                    'hidden_factor': rec.get('hidden_factor', 0.0)
                }
            )
            for rec in raw_recommendations
        ]
        
        response_time = time.time() - start_time
        
        return RecommendationResponse(
            user_id=user_id or "anonymous",
            session_id=f"popular_{int(time.time())}",
            items=items,
            ab_test_variant=None,
            ab_test_experiment=None,
            response_time=response_time,
            personalized=False,
            method="popular_static"
        )
        
    except Exception as e:
        logger.error(f"❌ Popular recommendations error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get popular recommendations: {str(e)}"
        )


# ═══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def _apply_variant_logic(
    recommendations: List[Dict],
    variant: str,
    limit: int
) -> List[Dict]:
    """
    Apply A/B test variant-specific logic to recommendations
    
    Variants:
    - control: No change (baseline)
    - diversity_boost: Increase variety of types/locations
    - popularity_weighted: Boost popular items
    - exploration: Include more novel items
    """
    if variant == "diversity_boost":
        # Ensure diverse types and locations
        seen_types = set()
        seen_locations = set()
        diverse_recs = []
        
        for rec in recommendations:
            rec_type = rec.get('type', 'unknown')
            rec_location = rec.get('neighborhood', rec.get('location', 'unknown'))
            
            # Prefer items with new type/location
            is_novel = rec_type not in seen_types or rec_location not in seen_locations
            
            if is_novel or len(diverse_recs) < limit // 2:
                diverse_recs.append(rec)
                seen_types.add(rec_type)
                seen_locations.add(rec_location)
            
            if len(diverse_recs) >= limit:
                break
        
        # Fill remaining with top-scored
        if len(diverse_recs) < limit:
            diverse_recs.extend(recommendations[len(diverse_recs):limit])
        
        return diverse_recs[:limit]
    
    elif variant == "popularity_weighted":
        # Boost items with high hidden_factor (popularity proxy)
        for rec in recommendations:
            original_score = rec.get('_personalization_score', 0.0)
            hidden_factor = rec.get('hidden_factor', 0.5)
            rec['_personalization_score'] = original_score * (0.7 + 0.3 * hidden_factor)
        
        recommendations.sort(
            key=lambda x: x.get('_personalization_score', 0.0),
            reverse=True
        )
        return recommendations[:limit]
    
    elif variant == "exploration":
        # Include more low-score items for exploration
        high_score = recommendations[:int(limit * 0.7)]
        exploration = recommendations[int(limit * 0.7):limit * 2]
        
        # Randomly sample from exploration set
        import random
        exploration_sample = random.sample(
            exploration,
            min(limit - len(high_score), len(exploration))
        )
        
        return high_score + exploration_sample
    
    else:
        # Unknown variant or control
        return recommendations[:limit]


def _interaction_to_reward(interaction_type: str) -> float:
    """
    Convert interaction type to numerical reward for learning
    
    Higher values = stronger positive signal
    """
    rewards = {
        "view": 0.1,      # Weak signal (just viewed)
        "click": 0.3,     # Moderate signal (engaged)
        "like": 0.7,      # Strong signal (liked it)
        "save": 0.8,      # Very strong (wants to remember)
        "share": 0.9,     # Strongest (recommending to others)
        "dismiss": -0.2,  # Negative signal
        "report": -1.0    # Strong negative signal
    }
    return rewards.get(interaction_type.lower(), 0.0)
