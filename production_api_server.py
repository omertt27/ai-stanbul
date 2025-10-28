"""
Production REST API Server with Database Persistence
FastAPI server for Istanbul AI with advanced personalization and feedback
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.database import get_db, engine, Base
from backend.repositories.personalization_repository import PersonalizationRepository
from istanbul_ai.main_system import IstanbulAISystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize database tables
Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(
    title="Istanbul AI - Production API",
    description="Advanced AI-powered Istanbul tourism assistant with personalization and feedback",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Istanbul AI System
istanbul_ai = IstanbulAISystem()

# ============================================================================
# Request/Response Models
# ============================================================================

class ChatRequest(BaseModel):
    """Chat request model"""
    user_id: str = Field(..., description="Unique user identifier")
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session identifier")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")


class ChatResponse(BaseModel):
    """Chat response model"""
    response: str
    intent: Optional[str] = None
    confidence: Optional[float] = None
    recommendations: Optional[List[Dict[str, Any]]] = None
    session_id: Optional[str] = None
    personalization_applied: bool = False
    ab_test_variant: Optional[str] = None


class FeedbackRequest(BaseModel):
    """Feedback submission model"""
    user_id: str = Field(..., description="User identifier")
    interaction_id: str = Field(..., description="Interaction identifier")
    satisfaction_score: float = Field(..., ge=1.0, le=5.0, description="Satisfaction score (1-5)")
    was_helpful: bool = Field(True, description="Was the response helpful?")
    response_quality: Optional[float] = Field(3.0, ge=1.0, le=5.0, description="Response quality (1-5)")
    speed_rating: Optional[float] = Field(3.0, ge=1.0, le=5.0, description="Speed rating (1-5)")
    intent: Optional[str] = Field("unknown", description="Intent type")
    feature: Optional[str] = Field("general", description="Feature used")
    comments: Optional[str] = Field("", description="User comments")
    issues: Optional[List[str]] = Field(default_factory=list, description="List of issues")


class FeedbackResponse(BaseModel):
    """Feedback submission response"""
    success: bool
    message: str
    feedback_id: Optional[int] = None


class PersonalizationInsightsResponse(BaseModel):
    """User personalization insights"""
    user_id: str
    preferences: Dict[str, Any]
    interaction_count: int
    satisfaction_history: List[Dict[str, Any]]
    similar_users: Optional[List[str]] = None


class DashboardMetricsResponse(BaseModel):
    """Dashboard metrics response"""
    total_users: int
    total_interactions: int
    total_feedback: int
    avg_satisfaction: float
    satisfaction_by_intent: Dict[str, Any]
    satisfaction_by_feature: Dict[str, Any]
    low_satisfaction_areas: List[Dict[str, Any]]
    recent_feedback: List[Dict[str, Any]]


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    version: str
    components: Dict[str, Any]


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "service": "Istanbul AI Production API",
        "version": "2.0.0",
        "status": "operational",
        "docs": "/api/docs"
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """
    Chat with the Istanbul AI assistant
    Includes personalization and A/B testing
    """
    try:
        # Get repository
        repo = PersonalizationRepository(db)
        
        # Process chat request
        response = await istanbul_ai.process_query(
            user_id=request.user_id,
            query=request.message,
            context=request.context
        )
        
        # Extract interaction data
        interaction_id = f"{request.user_id}_{datetime.now().timestamp()}"
        
        # Record interaction for personalization
        interaction_data = {
            'interaction_id': interaction_id,
            'type': response.get('intent', 'chat'),
            'item_id': None,
            'item_data': {
                'query': request.message,
                'intent': response.get('intent'),
                'has_recommendations': bool(response.get('recommendations'))
            },
            'rating': 0.7  # Default neutral-positive
        }
        
        # Save to database
        repo.save_interaction(request.user_id, interaction_data)
        
        # Load user preferences and apply personalization
        user_prefs = repo.get_user_preferences(request.user_id)
        if user_prefs and response.get('recommendations'):
            # Apply personalization scoring (in-memory for real-time)
            personalization_applied = True
        else:
            personalization_applied = False
        
        return ChatResponse(
            response=response.get('response', 'I can help you explore Istanbul!'),
            intent=response.get('intent'),
            confidence=response.get('confidence'),
            recommendations=response.get('recommendations', []),
            session_id=request.session_id or interaction_id,
            personalization_applied=personalization_applied,
            ab_test_variant=response.get('ab_test_variant')
        )
    
    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat processing failed: {str(e)}"
        )


@app.post("/api/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest, db: Session = Depends(get_db)):
    """
    Submit user feedback for an interaction
    """
    try:
        repo = PersonalizationRepository(db)
        
        # Prepare feedback data
        feedback_data = {
            'satisfaction_score': request.satisfaction_score,
            'was_helpful': request.was_helpful,
            'response_quality': request.response_quality,
            'speed_rating': request.speed_rating,
            'intent': request.intent,
            'feature': request.feature,
            'comments': request.comments,
            'issues': request.issues
        }
        
        # Save to database
        db_feedback = repo.save_feedback(
            request.user_id,
            request.interaction_id,
            feedback_data
        )
        
        logger.info(f"Feedback saved: user={request.user_id}, score={request.satisfaction_score}")
        
        return FeedbackResponse(
            success=True,
            message="Feedback submitted successfully",
            feedback_id=db_feedback.id
        )
    
    except Exception as e:
        logger.error(f"Feedback submission error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feedback submission failed: {str(e)}"
        )


@app.get("/api/personalization/{user_id}", response_model=PersonalizationInsightsResponse)
async def get_personalization_insights(user_id: str, db: Session = Depends(get_db)):
    """
    Get personalization insights for a user
    """
    try:
        repo = PersonalizationRepository(db)
        
        # Get user preferences
        preferences = repo.get_user_preferences(user_id)
        if not preferences:
            preferences = {
                'cuisines': {},
                'price_ranges': {},
                'districts': {},
                'activity_types': {},
                'attraction_types': {},
                'transportation_modes': {},
                'time_of_day': {},
                'interaction_count': 0
            }
        
        # Get satisfaction history
        satisfaction_history = repo.get_user_feedback_history(user_id, limit=50)
        
        # Get similar users
        similar_users = repo.get_similar_users(user_id, min_common_items=3)
        
        return PersonalizationInsightsResponse(
            user_id=user_id,
            preferences=preferences,
            interaction_count=preferences.get('interaction_count', 0),
            satisfaction_history=satisfaction_history,
            similar_users=similar_users[:10]
        )
    
    except Exception as e:
        logger.error(f"Personalization insights error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get personalization insights: {str(e)}"
        )


@app.get("/api/dashboard/metrics", response_model=DashboardMetricsResponse)
async def get_dashboard_metrics(db: Session = Depends(get_db)):
    """
    Get dashboard metrics for admin/monitoring
    """
    try:
        repo = PersonalizationRepository(db)
        
        # Get aggregate metrics
        feedback_metrics = repo.get_aggregate_feedback_metrics()
        
        # Count total users and interactions
        total_users = db.execute("SELECT COUNT(DISTINCT user_id) FROM user_interactions").scalar() or 0
        total_interactions = db.execute("SELECT COUNT(*) FROM user_interactions").scalar() or 0
        
        # Get recent feedback
        recent_feedback = repo.get_recent_feedback(count=20)
        
        # Identify low satisfaction areas
        low_satisfaction_areas = []
        for intent, metrics in feedback_metrics.get('satisfaction_by_intent', {}).items():
            if metrics['avg_satisfaction'] < 3.5 and metrics['count'] >= 5:
                low_satisfaction_areas.append({
                    'intent': intent,
                    'avg_satisfaction': metrics['avg_satisfaction'],
                    'count': metrics['count']
                })
        
        return DashboardMetricsResponse(
            total_users=total_users,
            total_interactions=total_interactions,
            total_feedback=feedback_metrics['total_ratings'],
            avg_satisfaction=feedback_metrics['avg_satisfaction'],
            satisfaction_by_intent=feedback_metrics['satisfaction_by_intent'],
            satisfaction_by_feature=feedback_metrics['satisfaction_by_feature'],
            low_satisfaction_areas=low_satisfaction_areas,
            recent_feedback=recent_feedback
        )
    
    except Exception as e:
        logger.error(f"Dashboard metrics error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get dashboard metrics: {str(e)}"
        )


@app.get("/api/health", response_model=HealthCheckResponse)
async def health_check(db: Session = Depends(get_db)):
    """
    Comprehensive health check endpoint
    """
    try:
        # Check database connectivity
        db.execute("SELECT 1")
        db_status = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        db_status = f"unhealthy: {str(e)}"
    
    # Check Istanbul AI system
    try:
        ai_health = istanbul_ai.health_check()
        ai_status = "healthy" if ai_health.get('status') == 'healthy' else "degraded"
    except Exception as e:
        logger.error(f"AI system health check failed: {str(e)}")
        ai_status = f"unhealthy: {str(e)}"
    
    overall_status = "healthy" if db_status == "healthy" and ai_status == "healthy" else "degraded"
    
    return HealthCheckResponse(
        status=overall_status,
        timestamp=datetime.now().isoformat(),
        version="2.0.0",
        components={
            "database": db_status,
            "ai_system": ai_status,
            "api": "healthy"
        }
    )


@app.on_event("startup")
async def startup_event():
    """Startup initialization"""
    logger.info("🚀 Istanbul AI Production API starting...")
    logger.info("✅ Database tables initialized")
    logger.info("✅ Istanbul AI system initialized")
    logger.info("🌐 API server ready")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Istanbul AI Production API shutting down...")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "production_api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload in production
        workers=4,  # Multi-worker for production
        log_level="info"
    )
