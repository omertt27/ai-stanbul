"""
Monitoring API endpoints for Week 3-4
Custom metrics dashboard API (Vercel/Render compatible)
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from datetime import datetime, timedelta
import logging

from backend.services.realtime_feedback_loop import get_realtime_feedback_loop
from backend.database import SessionLocal
from backend.models import FeedbackEvent, UserInteractionAggregate
from sqlalchemy import func, distinct

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        loop = get_realtime_feedback_loop()
        metrics = loop.get_metrics()
        
        # Check cache connection
        cache_connected = metrics.get('cache_stats', {}).get('connected', False)
        
        return {
            'status': 'healthy' if cache_connected else 'degraded',
            'timestamp': datetime.now().isoformat(),
            'services': {
                'feedback_loop': loop.is_running,
                'redis_cache': cache_connected,
                'database': True  # Will fail if DB is down
            }
        }
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/metrics")
async def get_metrics():
    """Get comprehensive system metrics"""
    try:
        loop = get_realtime_feedback_loop()
        metrics = loop.get_metrics()
        
        # Get database stats
        db_stats = await _get_database_stats()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'feedback_collector': metrics.get('collector_metrics', {}),
            'online_learning': metrics.get('learning_engine_metrics', {}),
            'cache': metrics.get('cache_stats', {}),
            'database': db_stats,
            'system': {
                'is_running': metrics.get('is_running', False)
            }
        }
    except Exception as e:
        logger.error(f"❌ Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/feedback")
async def get_feedback_metrics(hours: int = 24):
    """Get feedback event metrics for the last N hours"""
    try:
        db = SessionLocal()
        
        since = datetime.now() - timedelta(hours=hours)
        
        # Total events
        total_events = db.query(func.count(FeedbackEvent.id)).filter(
            FeedbackEvent.timestamp >= since
        ).scalar()
        
        # Events by type
        events_by_type = db.query(
            FeedbackEvent.event_type,
            func.count(FeedbackEvent.id).label('count')
        ).filter(
            FeedbackEvent.timestamp >= since
        ).group_by(FeedbackEvent.event_type).all()
        
        # Unique users
        unique_users = db.query(
            func.count(distinct(FeedbackEvent.user_id))
        ).filter(
            FeedbackEvent.timestamp >= since
        ).scalar()
        
        # Unique items
        unique_items = db.query(
            func.count(distinct(FeedbackEvent.item_id))
        ).filter(
            FeedbackEvent.timestamp >= since
        ).scalar()
        
        db.close()
        
        return {
            'period_hours': hours,
            'since': since.isoformat(),
            'total_events': total_events,
            'events_by_type': {
                event_type: count for event_type, count in events_by_type
            },
            'unique_users': unique_users,
            'unique_items': unique_items,
            'avg_events_per_user': round(total_events / unique_users, 2) if unique_users > 0 else 0
        }
    except Exception as e:
        logger.error(f"❌ Failed to get feedback metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/recommendations")
async def get_recommendation_metrics():
    """Get recommendation serving metrics"""
    try:
        loop = get_realtime_feedback_loop()
        metrics = loop.get_metrics()
        
        learning_metrics = metrics.get('learning_engine_metrics', {})
        cache_stats = metrics.get('cache_stats', {})
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_predictions': learning_metrics.get('total_predictions', 0),
            'total_updates': learning_metrics.get('total_updates', 0),
            'avg_prediction_time_ms': learning_metrics.get('avg_prediction_time', 0) * 1000,
            'avg_update_time_ms': learning_metrics.get('avg_update_time', 0) * 1000,
            'cache': {
                'hit_rate_percent': cache_stats.get('hit_rate', 0),
                'total_keys': cache_stats.get('keys_count', 0),
                'memory_mb': cache_stats.get('used_memory_mb', 0)
            }
        }
    except Exception as e:
        logger.error(f"❌ Failed to get recommendation metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/engagement")
async def get_engagement_metrics(hours: int = 24):
    """Get user engagement metrics"""
    try:
        db = SessionLocal()
        
        since = datetime.now() - timedelta(hours=hours)
        
        # Get all aggregates updated in the period
        aggregates = db.query(UserInteractionAggregate).filter(
            UserInteractionAggregate.last_interaction >= since
        ).all()
        
        if not aggregates:
            db.close()
            return {
                'period_hours': hours,
                'active_users': 0,
                'avg_ctr': 0,
                'avg_conversion_rate': 0,
                'avg_rating': 0
            }
        
        # Calculate aggregated metrics
        total_users = len(aggregates)
        avg_ctr = sum(agg.click_through_rate for agg in aggregates) / total_users
        
        # Only count aggregates with clicks for conversion rate
        aggregates_with_clicks = [agg for agg in aggregates if agg.click_count > 0]
        avg_conversion = (
            sum(agg.conversion_rate for agg in aggregates_with_clicks) / len(aggregates_with_clicks)
            if aggregates_with_clicks else 0
        )
        
        # Only count aggregates with ratings
        aggregates_with_ratings = [agg for agg in aggregates if agg.rating_count > 0]
        avg_rating = (
            sum(agg.avg_rating for agg in aggregates_with_ratings) / len(aggregates_with_ratings)
            if aggregates_with_ratings else 0
        )
        
        db.close()
        
        return {
            'period_hours': hours,
            'since': since.isoformat(),
            'active_users': total_users,
            'avg_click_through_rate': round(avg_ctr, 4),
            'avg_conversion_rate': round(avg_conversion, 4),
            'avg_rating': round(avg_rating, 2),
            'users_with_ratings': len(aggregates_with_ratings),
            'users_with_conversions': len(aggregates_with_clicks)
        }
    except Exception as e:
        logger.error(f"❌ Failed to get engagement metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/cache")
async def get_cache_metrics():
    """Get detailed cache statistics"""
    try:
        loop = get_realtime_feedback_loop()
        cache_stats = loop.redis_cache.get_stats()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'connected': cache_stats.get('connected', False),
            'hits': cache_stats.get('hits', 0),
            'misses': cache_stats.get('misses', 0),
            'hit_rate_percent': cache_stats.get('hit_rate', 0),
            'keys_count': cache_stats.get('keys_count', 0),
            'memory_used_mb': cache_stats.get('used_memory_mb', 0),
            'uptime_days': cache_stats.get('uptime_days', 0)
        }
    except Exception as e:
        logger.error(f"❌ Failed to get cache metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/clear")
async def clear_cache(pattern: str = None):
    """Clear cache (admin only - add auth in production!)"""
    try:
        loop = get_realtime_feedback_loop()
        
        if pattern:
            loop.redis_cache.clear_pattern(pattern)
            return {
                'status': 'success',
                'message': f'Cleared cache keys matching pattern: {pattern}'
            }
        else:
            loop.redis_cache.clear_all()
            return {
                'status': 'success',
                'message': 'Cleared all cache'
            }
    except Exception as e:
        logger.error(f"❌ Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _get_database_stats() -> Dict[str, Any]:
    """Get database statistics"""
    try:
        db = SessionLocal()
        
        # Count total events
        total_events = db.query(func.count(FeedbackEvent.id)).scalar()
        
        # Count total aggregates
        total_aggregates = db.query(func.count(UserInteractionAggregate.id)).scalar()
        
        # Count unique users
        unique_users = db.query(func.count(distinct(FeedbackEvent.user_id))).scalar()
        
        db.close()
        
        return {
            'total_feedback_events': total_events,
            'total_user_aggregates': total_aggregates,
            'unique_users': unique_users
        }
    except Exception as e:
        logger.error(f"❌ Failed to get database stats: {e}")
        return {
            'error': str(e)
        }


# Dashboard data endpoint
@router.get("/dashboard")
async def get_dashboard_data(hours: int = 24):
    """Get all metrics for dashboard visualization"""
    try:
        # Gather all metrics in parallel
        metrics = await get_metrics()
        feedback_metrics = await get_feedback_metrics(hours)
        rec_metrics = await get_recommendation_metrics()
        engagement_metrics = await get_engagement_metrics(hours)
        cache_metrics = await get_cache_metrics()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'period_hours': hours,
            'summary': {
                'total_events': feedback_metrics['total_events'],
                'active_users': engagement_metrics['active_users'],
                'cache_hit_rate': cache_metrics['hit_rate_percent'],
                'avg_rating': engagement_metrics['avg_rating']
            },
            'feedback': feedback_metrics,
            'recommendations': rec_metrics,
            'engagement': engagement_metrics,
            'cache': cache_metrics,
            'system': metrics
        }
    except Exception as e:
        logger.error(f"❌ Failed to get dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e))
