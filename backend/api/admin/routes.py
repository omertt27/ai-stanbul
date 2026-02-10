"""
Comprehensive Admin API Routes
Aggregates all admin dashboard endpoints for centralized management
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
import json
import io
import csv

from database import get_db, SessionLocal
from models import (
    BlogPost as BlogPostModel,
    # IntentFeedback,  # Temporarily disabled - model doesn't exist
    FeedbackEvent,
    UserInteractionAggregate
)
from sqlalchemy import func, distinct, desc

logger = logging.getLogger(__name__)

# Create main admin router
# Note: Prefix is added during router registration in main_modular.py
router = APIRouter(tags=["admin"])


# ============================================================================
# STATS & OVERVIEW ENDPOINTS
# ============================================================================

@router.get("/stats")
async def get_admin_stats(db: Session = Depends(get_db)):
    """Get overall admin dashboard statistics"""
    try:
        # Get counts from database
        total_posts = db.query(func.count(BlogPostModel.id)).scalar() or 0
        
        # Get feedback stats from last 7 days
        since = datetime.now() - timedelta(days=7)
        recent_feedback = db.query(func.count(FeedbackEvent.id)).filter(
            FeedbackEvent.timestamp >= since
        ).scalar() or 0
        
        # Get intent feedback stats
        # intent_feedback_count = db.query(func.count(IntentFeedback.id)).filter(
        #     IntentFeedback.created_at >= since
        # ).scalar() or 0
        intent_feedback_count = 0  # Temporarily disabled - IntentFeedback model doesn't exist
        
        # Get unique users
        unique_users = db.query(
            func.count(distinct(FeedbackEvent.user_id))
        ).filter(
            FeedbackEvent.timestamp >= since
        ).scalar() or 0
        
        return {
            "status": "success",
            "data": {
                "blog_posts": total_posts,
                "recent_feedback": recent_feedback,
                "intent_feedback": intent_feedback_count,
                "active_users": unique_users,
                "last_updated": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error getting admin stats: {e}")
        return {
            "status": "success",
            "data": {
                "blog_posts": 0,
                "recent_feedback": 0,
                "intent_feedback": 0,
                "active_users": 0,
                "last_updated": datetime.now().isoformat()
            }
        }


# ============================================================================
# BLOG MANAGEMENT ENDPOINTS
# ============================================================================

@router.get("/blog/posts")
async def get_blog_posts(
    status: Optional[str] = None,
    limit: int = Query(50, le=100),
    offset: int = Query(0),
    db: Session = Depends(get_db)
):
    """Get all blog posts with optional filtering"""
    try:
        query = db.query(BlogPostModel)
        
        if status:
            query = query.filter(BlogPostModel.status == status)
        
        query = query.order_by(desc(BlogPostModel.created_at))
        total = query.count()
        posts = query.offset(offset).limit(limit).all()
        
        return {
            "status": "success",
            "data": {
                "posts": [
                    {
                        "id": post.id,
                        "title": post.title,
                        "slug": post.slug,
                        "excerpt": post.excerpt,
                        "content": post.content,
                        "author": post.author,
                        "status": post.status,
                        "featured_image": post.featured_image,
                        "category": post.category,
                        "tags": post.tags,
                        "views": post.views or 0,
                        "likes": post.likes or 0,
                        "created_at": post.created_at.isoformat() if post.created_at else None,
                        "updated_at": post.updated_at.isoformat() if post.updated_at else None,
                        "published_at": post.published_at.isoformat() if post.published_at else None
                    }
                    for post in posts
                ],
                "total": total,
                "limit": limit,
                "offset": offset
            }
        }
    except Exception as e:
        logger.error(f"Error getting blog posts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/blog/posts")
async def create_blog_post(
    post_data: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db)
):
    """Create a new blog post"""
    try:
        new_post = BlogPostModel(
            title=post_data.get("title"),
            slug=post_data.get("slug", "").lower().replace(" ", "-"),
            content=post_data.get("content"),
            excerpt=post_data.get("excerpt", ""),
            author=post_data.get("author", "Admin"),
            status=post_data.get("status", "draft"),
            featured_image=post_data.get("featured_image"),
            category=post_data.get("category"),
            tags=post_data.get("tags", []),
            created_at=datetime.now()
        )
        
        db.add(new_post)
        db.commit()
        db.refresh(new_post)
        
        return {
            "status": "success",
            "data": {
                "id": new_post.id,
                "message": "Blog post created successfully"
            }
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating blog post: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/blog/posts/{post_id}")
async def delete_blog_post(post_id: int, db: Session = Depends(get_db)):
    """Delete a blog post"""
    try:
        post = db.query(BlogPostModel).filter(BlogPostModel.id == post_id).first()
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")
        
        db.delete(post)
        db.commit()
        
        return {
            "status": "success",
            "message": "Blog post deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting blog post: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# COMMENTS MANAGEMENT ENDPOINTS
# ============================================================================

@router.get("/comments")
async def get_comments(
    status: Optional[str] = None,
    limit: int = Query(50, le=100),
    db: Session = Depends(get_db)
):
    """Get all comments with optional filtering"""
    try:
        # For now, return empty list as comments model may not be fully set up
        # This can be enhanced when comments model is available
        return {
            "status": "success",
            "data": {
                "comments": [],
                "total": 0,
                "message": "Comments feature coming soon"
            }
        }
    except Exception as e:
        logger.error(f"Error getting comments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# FEEDBACK MANAGEMENT ENDPOINTS
# ============================================================================

@router.get("/feedback/export")
async def export_feedback(
    days: int = Query(30, le=365),
    format: str = Query("json", pattern="^(json|csv)$"),
    db: Session = Depends(get_db)
):
    """Export feedback data in JSON or CSV format"""
    try:
        since = datetime.now() - timedelta(days=days)
        
        # Get feedback events
        feedback_events = db.query(FeedbackEvent).filter(
            FeedbackEvent.timestamp >= since
        ).order_by(desc(FeedbackEvent.timestamp)).all()
        
        feedback_data = [
            {
                "id": event.id,
                "user_id": event.user_id,
                "session_id": event.session_id,
                "event_type": event.event_type,
                "item_id": event.item_id,
                "rating": event.rating,
                "feedback_text": event.feedback_text,
                "context": event.context,
                "timestamp": event.timestamp.isoformat() if event.timestamp else None
            }
            for event in feedback_events
        ]
        
        if format == "csv":
            # Create CSV response
            output = io.StringIO()
            if feedback_data:
                writer = csv.DictWriter(output, fieldnames=feedback_data[0].keys())
                writer.writeheader()
                writer.writerows(feedback_data)
            
            return StreamingResponse(
                iter([output.getvalue()]),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=feedback_export_{datetime.now().strftime('%Y%m%d')}.csv"
                }
            )
        else:
            # Return JSON
            return {
                "status": "success",
                "data": {
                    "feedback": feedback_data,
                    "total": len(feedback_data),
                    "period_days": days,
                    "exported_at": datetime.now().isoformat()
                }
            }
    except Exception as e:
        logger.error(f"Error exporting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ANALYTICS ENDPOINTS
# ============================================================================

@router.get("/analytics")
async def get_analytics(
    days: int = Query(7, le=90),
    db: Session = Depends(get_db)
):
    """Get analytics data for the specified time period"""
    try:
        since = datetime.now() - timedelta(days=days)
        
        # Get daily feedback counts
        daily_feedback = db.query(
            func.date(FeedbackEvent.timestamp).label('date'),
            func.count(FeedbackEvent.id).label('count')
        ).filter(
            FeedbackEvent.timestamp >= since
        ).group_by(func.date(FeedbackEvent.timestamp)).all()
        
        # Get feedback by type
        feedback_by_type = db.query(
            FeedbackEvent.event_type,
            func.count(FeedbackEvent.id).label('count')
        ).filter(
            FeedbackEvent.timestamp >= since
        ).group_by(FeedbackEvent.event_type).all()
        
        # Get unique users per day
        daily_users = db.query(
            func.date(FeedbackEvent.timestamp).label('date'),
            func.count(distinct(FeedbackEvent.user_id)).label('users')
        ).filter(
            FeedbackEvent.timestamp >= since
        ).group_by(func.date(FeedbackEvent.timestamp)).all()
        
        return {
            "status": "success",
            "data": {
                "period_days": days,
                "daily_feedback": [
                    {
                        "date": str(row.date),
                        "count": row.count
                    }
                    for row in daily_feedback
                ],
                "feedback_by_type": {
                    row.event_type: row.count
                    for row in feedback_by_type
                },
                "daily_active_users": [
                    {
                        "date": str(row.date),
                        "users": row.users
                    }
                    for row in daily_users
                ],
                "generated_at": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# INTENT CLASSIFICATION ENDPOINTS
# ============================================================================

@router.get("/intents/stats")
async def get_intent_stats(
    days: int = Query(7, le=90),
    db: Session = Depends(get_db)
):
    """Get intent classification statistics"""
    try:
        since = datetime.now() - timedelta(days=days)
        
        # Temporarily disabled - IntentFeedback model doesn't exist
        # Get total intent feedback
        # total_feedback = db.query(func.count(IntentFeedback.id)).filter(
        #     IntentFeedback.created_at >= since
        # ).scalar() or 0
        total_feedback = 0
        
        # Get feedback by correctness
        # correct_count = db.query(func.count(IntentFeedback.id)).filter(
        #     IntentFeedback.created_at >= since,
        #     IntentFeedback.is_correct == True
        # ).scalar() or 0
        correct_count = 0
        
        # incorrect_count = db.query(func.count(IntentFeedback.id)).filter(
        #     IntentFeedback.created_at >= since,
        #     IntentFeedback.is_correct == False
        # ).scalar() or 0
        incorrect_count = 0
        
        # Get average confidence
        # avg_confidence = db.query(func.avg(IntentFeedback.predicted_confidence)).filter(
        #     IntentFeedback.created_at >= since
        # ).scalar() or 0.0
        avg_confidence = 0.0
        
        # Get intent distribution
        # intent_distribution = db.query(
        #     IntentFeedback.predicted_intent,
        #     func.count(IntentFeedback.id).label('count')
        # ).filter(
        #     IntentFeedback.created_at >= since
        # ).group_by(IntentFeedback.predicted_intent).all()
        intent_distribution = []
        
        accuracy = (correct_count / total_feedback * 100) if total_feedback > 0 else 0
        
        return {
            "status": "success",
            "data": {
                "total_classifications": total_feedback,
                "correct": correct_count,
                "incorrect": incorrect_count,
                "accuracy_percent": round(accuracy, 2),
                "average_confidence": round(float(avg_confidence), 4),
                "intent_distribution": {
                    row.predicted_intent: row.count
                    for row in intent_distribution
                },
                "period_days": days,
                "generated_at": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error getting intent stats: {e}")
        return {
            "status": "success",
            "data": {
                "total_classifications": 0,
                "correct": 0,
                "incorrect": 0,
                "accuracy_percent": 0,
                "average_confidence": 0,
                "intent_distribution": {},
                "period_days": days,
                "generated_at": datetime.now().isoformat()
            }
        }


# ============================================================================
# SYSTEM METRICS ENDPOINTS
# ============================================================================

@router.get("/system/metrics")
async def get_system_metrics(
    hours: int = Query(24, le=168),
    db: Session = Depends(get_db)
):
    """Get system performance metrics"""
    try:
        since = datetime.now() - timedelta(hours=hours)
        
        # Get feedback metrics
        total_events = db.query(func.count(FeedbackEvent.id)).filter(
            FeedbackEvent.timestamp >= since
        ).scalar() or 0
        
        # Get unique users and sessions
        unique_users = db.query(
            func.count(distinct(FeedbackEvent.user_id))
        ).filter(
            FeedbackEvent.timestamp >= since
        ).scalar() or 0
        
        unique_sessions = db.query(
            func.count(distinct(FeedbackEvent.session_id))
        ).filter(
            FeedbackEvent.timestamp >= since
        ).scalar() or 0
        
        # Get hourly event distribution
        hourly_events = db.query(
            func.date_trunc('hour', FeedbackEvent.timestamp).label('hour'),
            func.count(FeedbackEvent.id).label('count')
        ).filter(
            FeedbackEvent.timestamp >= since
        ).group_by(func.date_trunc('hour', FeedbackEvent.timestamp)).all()
        
        return {
            "status": "success",
            "data": {
                "period_hours": hours,
                "total_events": total_events,
                "unique_users": unique_users,
                "unique_sessions": unique_sessions,
                "events_per_hour": round(total_events / hours, 2) if hours > 0 else 0,
                "hourly_distribution": [
                    {
                        "hour": row.hour.isoformat() if row.hour else None,
                        "count": row.count
                    }
                    for row in hourly_events
                ],
                "system_status": "healthy",
                "generated_at": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        return {
            "status": "success",
            "data": {
                "period_hours": hours,
                "total_events": 0,
                "unique_users": 0,
                "unique_sessions": 0,
                "events_per_hour": 0,
                "hourly_distribution": [],
                "system_status": "unknown",
                "generated_at": datetime.now().isoformat()
            }
        }


# ============================================================================
# HEALTH CHECK
# ============================================================================

@router.get("/health")
async def admin_health_check():
    """Health check for admin API"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "admin-api"
    }


logger.info("✅ Admin routes initialized")
