"""
LLM Statistics API Routes

Provides comprehensive analytics and monitoring endpoints for the
modular Pure LLM Handler system.

Endpoints:
- GET /api/v1/llm/stats - General statistics overview
- GET /api/v1/llm/stats/signals - Signal detection analytics
- GET /api/v1/llm/stats/performance - Performance metrics
- GET /api/v1/llm/stats/cache - Cache statistics
- GET /api/v1/llm/stats/users - User behavior analytics
- GET /api/v1/llm/stats/export - Export statistics
- WS /api/v1/llm/stats/stream - Real-time metrics stream

Author: AI Istanbul Team
Date: November 15, 2025
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, FileResponse
import json
import csv
import io

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/api/v1/llm",
    tags=["LLM Statistics"]
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"📊 WebSocket client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"📊 WebSocket client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to client: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()


# Dependency to get analytics manager from core
def get_analytics_manager():
    """
    Get analytics manager instance from the Pure LLM Core.
    This assumes pure_llm_core is available globally.
    """
    try:
        # Import at runtime to avoid circular dependencies
        import sys
        import os
        
        # Add parent directory to path if not already there
        backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)
        
        from main import pure_llm_core
        
        if not pure_llm_core:
            raise HTTPException(
                status_code=503,
                detail="Pure LLM Core not initialized. Please ensure the server has started properly."
            )
        
        if not hasattr(pure_llm_core, 'analytics') or not pure_llm_core.analytics:
            raise HTTPException(
                status_code=503,
                detail="Analytics system not available in Pure LLM Core"
            )
        
        return pure_llm_core.analytics
        
    except ImportError as e:
        logger.error(f"Failed to import pure_llm_core: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Failed to import Pure LLM Core: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error accessing analytics manager: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Error accessing analytics system: {str(e)}"
        )


# Authentication dependency (placeholder - implement your auth logic)
async def verify_admin_access():
    """
    Verify admin access for statistics endpoints.
    TODO: Implement proper authentication/authorization.
    """
    # For now, allow all access
    # In production, verify JWT token, API key, or session
    return True


@router.get("/stats")
async def get_general_stats(
    admin: bool = Depends(verify_admin_access),
    analytics: Any = Depends(get_analytics_manager)
) -> Dict[str, Any]:
    """
    Get general system statistics overview.
    
    Returns:
        Dictionary with comprehensive system statistics
    """
    try:
        summary = analytics.get_summary()
        
        # Calculate derived metrics
        total_queries = summary['basic_stats'].get('total_queries', 0)
        cache_hits = summary['basic_stats'].get('cache_hits', 0)
        cache_misses = summary['basic_stats'].get('cache_misses', 0)
        total_cache_requests = cache_hits + cache_misses
        
        cache_hit_rate = (cache_hits / total_cache_requests) if total_cache_requests > 0 else 0
        
        # Get performance stats
        perf = summary.get('performance', {})
        query_stats = perf.get('query_latency', {})
        
        # Get top signals
        signals = summary.get('signals', {})
        detections = signals.get('by_signal', {})
        top_signals = sorted(
            [{'signal': k, 'count': v} for k, v in detections.items()],
            key=lambda x: x['count'],
            reverse=True
        )[:5]
        
        # Get language distribution
        users = summary.get('users', {})
        languages = users.get('by_language', {})
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "total_queries": total_queries,
            "cache_hit_rate": round(cache_hit_rate, 3),
            "average_response_time_ms": round(query_stats.get('avg', 0) * 1000, 2),
            "error_rate": round(
                summary['errors'].get('total', 0) / max(total_queries, 1),
                4
            ),
            "active_users": users.get('unique_users', 0),
            "top_signals": top_signals,
            "languages": dict(languages),
            "performance": {
                "p50_ms": round(query_stats.get('p50', 0) * 1000, 2),
                "p95_ms": round(query_stats.get('p95', 0) * 1000, 2),
                "p99_ms": round(query_stats.get('p99', 0) * 1000, 2)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting general stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/signals")
async def get_signal_stats(
    date_from: Optional[str] = Query(None, description="Start date (ISO format)"),
    date_to: Optional[str] = Query(None, description="End date (ISO format)"),
    signal_type: Optional[str] = Query(None, description="Filter by signal type"),
    language: Optional[str] = Query(None, description="Filter by language"),
    admin: bool = Depends(verify_admin_access),
    analytics: Any = Depends(get_analytics_manager)
) -> Dict[str, Any]:
    """
    Get detailed signal detection analytics.
    
    Args:
        date_from: Start date for filtering
        date_to: End date for filtering
        signal_type: Specific signal to analyze
        language: Filter by language
    
    Returns:
        Detailed signal detection statistics
    """
    try:
        summary = analytics.get_summary()
        signals_data = summary.get('signals', {})
        users_data = summary.get('users', {})
        
        # Get signal detections
        by_signal = signals_data.get('by_signal', {})
        total_queries = summary['basic_stats'].get('total_queries', 1)
        
        # Filter by signal_type if specified
        if signal_type:
            by_signal = {signal_type: by_signal.get(signal_type, 0)}
        
        # Build detailed signal stats
        signals_detected = {}
        for sig_name, count in by_signal.items():
            signals_detected[sig_name] = {
                "count": count,
                "percentage": round((count / total_queries) * 100, 2) if total_queries > 0 else 0,
                "avg_confidence": 0.85,  # TODO: Calculate from confidence scores
                "languages": {}  # TODO: Break down by language
            }
        
        return {
            "period": {
                "from": date_from or (datetime.now() - timedelta(days=1)).isoformat(),
                "to": date_to or datetime.now().isoformat()
            },
            "total_queries": total_queries,
            "signals_detected": signals_detected,
            "multi_intent_queries": signals_data.get('multi_intent', 0),
            "zero_signal_queries": total_queries - sum(by_signal.values()),
            "signal_combinations": []  # TODO: Track signal combinations
        }
        
    except Exception as e:
        logger.error(f"Error getting signal stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/performance")
async def get_performance_stats(
    date_from: Optional[str] = Query(None, description="Start date"),
    date_to: Optional[str] = Query(None, description="End date"),
    aggregation: str = Query("hour", description="Time aggregation (minute/hour/day)"),
    admin: bool = Depends(verify_admin_access),
    analytics: Any = Depends(get_analytics_manager)
) -> Dict[str, Any]:
    """
    Get system performance metrics.
    
    Args:
        date_from: Start date for analysis
        date_to: End date for analysis
        aggregation: Time bucket size (minute/hour/day)
    
    Returns:
        Performance statistics and time-series data
    """
    try:
        summary = analytics.get_summary()
        perf = summary.get('performance', {})
        
        # Query latency stats
        query_latency = perf.get('query_latency', {})
        llm_latency = perf.get('llm_latency', {})
        
        # Convert to milliseconds
        latency_stats = {
            "p50": round(query_latency.get('p50', 0) * 1000, 2),
            "p90": round(query_latency.get('p90', 0) * 1000, 2),
            "p95": round(query_latency.get('p95', 0) * 1000, 2),
            "p99": round(query_latency.get('p99', 0) * 1000, 2),
            "avg": round(query_latency.get('avg', 0) * 1000, 2),
            "min": round(query_latency.get('min', 0) * 1000, 2),
            "max": round(query_latency.get('max', 0) * 1000, 2)
        }
        
        # Latency breakdown (estimated)
        avg_query = query_latency.get('avg', 0) * 1000
        avg_llm = llm_latency.get('avg', 0) * 1000
        
        breakdown = {
            "signal_detection_ms": {
                "avg": 45,
                "p95": 80
            },
            "context_building_ms": {
                "avg": max(0, avg_query - avg_llm - 45),
                "p95": 350
            },
            "llm_generation_ms": {
                "avg": round(avg_llm, 2),
                "p95": round(llm_latency.get('p95', 0) * 1000, 2)
            },
            "total_processing_ms": {
                "avg": round(avg_query, 2),
                "p95": round(query_latency.get('p95', 0) * 1000, 2)
            }
        }
        
        # Get hourly trends
        hourly_trends = summary.get('hourly_trends', [])
        time_series = []
        for trend_item in hourly_trends:
            time_series.append({
                "timestamp": trend_item.get('timestamp', ''),
                "queries": trend_item.get('queries', 0),
                "avg_latency": round(trend_item.get('avg_latency', 0) * 1000, 2),
                "error_rate": round(
                    trend_item.get('errors', 0) / max(trend_item.get('queries', 1), 1),
                    4
                )
            })
        
        # Identify bottlenecks
        bottlenecks = []
        if avg_llm > 0 and avg_query > 0:
            llm_percentage = (avg_llm / avg_query) * 100
            if llm_percentage > 60:
                bottlenecks.append({
                    "component": "llm_generation",
                    "impact": "high",
                    "percentage": round(llm_percentage, 1)
                })
        
        return {
            "period": {
                "from": date_from or (datetime.now() - timedelta(hours=24)).isoformat(),
                "to": date_to or datetime.now().isoformat(),
                "aggregation": aggregation
            },
            "latency": latency_stats,
            "breakdown": breakdown,
            "time_series": time_series[-24:],  # Last 24 hours/periods
            "bottlenecks": bottlenecks
        }
        
    except Exception as e:
        logger.error(f"Error getting performance stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/cache")
async def get_cache_stats(
    admin: bool = Depends(verify_admin_access),
    analytics: Any = Depends(get_analytics_manager)
) -> Dict[str, Any]:
    """
    Get cache performance metrics.
    
    Returns:
        Cache statistics and efficiency metrics
    """
    try:
        summary = analytics.get_summary()
        
        cache_hits = summary['basic_stats'].get('cache_hits', 0)
        cache_misses = summary['basic_stats'].get('cache_misses', 0)
        total_requests = cache_hits + cache_misses
        
        hit_rate = (cache_hits / total_requests) if total_requests > 0 else 0
        miss_rate = 1 - hit_rate
        
        # Get cache latency stats
        perf = summary.get('performance', {})
        cache_latency = perf.get('cache_latency', {})
        
        return {
            "status": "healthy",
            "cache_type": "redis",
            "statistics": {
                "total_requests": total_requests,
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "hit_rate": round(hit_rate, 3),
                "miss_rate": round(miss_rate, 3),
                "avg_cache_lookup_ms": round(cache_latency.get('avg', 0) * 1000, 2),
                "semantic_cache": {
                    "enabled": True,
                    "threshold": 0.85,
                    "hits": 0,  # TODO: Track separately
                    "hit_rate": 0.0
                },
                "exact_match_cache": {
                    "hits": cache_hits,
                    "hit_rate": round(hit_rate, 3)
                }
            },
            "memory": {
                "total_keys": 0,  # TODO: Get from Redis
                "memory_used_mb": 0,
                "evicted_keys": 0
            },
            "top_cached_queries": [],  # TODO: Track popular queries
            "ttl_distribution": {}  # TODO: Get from Redis
        }
        
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/users")
async def get_user_stats(
    date_from: Optional[str] = Query(None, description="Start date"),
    date_to: Optional[str] = Query(None, description="End date"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    admin: bool = Depends(verify_admin_access),
    analytics: Any = Depends(get_analytics_manager)
) -> Dict[str, Any]:
    """
    Get user behavior analytics.
    
    Args:
        date_from: Start date for analysis
        date_to: End date for analysis
        user_id: Filter by specific user
    
    Returns:
        User engagement and behavior statistics
    """
    try:
        summary = analytics.get_summary()
        users_data = summary.get('users', {})
        signals_data = summary.get('signals', {})
        
        unique_users = users_data.get('unique_users', 0)
        queries_by_language = users_data.get('by_language', {})
        queries_by_user = users_data.get('by_user', {})
        
        total_queries = summary['basic_stats'].get('total_queries', 0)
        avg_queries_per_user = total_queries / max(unique_users, 1)
        
        return {
            "period": {
                "from": date_from or (datetime.now() - timedelta(days=1)).isoformat(),
                "to": date_to or datetime.now().isoformat()
            },
            "unique_users": unique_users,
            "new_users": 0,  # TODO: Track new vs returning
            "returning_users": unique_users,
            "engagement": {
                "avg_queries_per_user": round(avg_queries_per_user, 2),
                "avg_session_duration_min": 0,  # TODO: Track sessions
                "bounce_rate": 0.0
            },
            "language_preferences": dict(queries_by_language),
            "popular_queries": [],  # TODO: Track popular queries
            "query_patterns": {
                "restaurant_searches": signals_data.get('by_signal', {}).get('needs_restaurant', 0),
                "attraction_searches": signals_data.get('by_signal', {}).get('needs_attraction', 0),
                "transportation_searches": signals_data.get('by_signal', {}).get('needs_transportation', 0),
                "accommodation_searches": signals_data.get('by_signal', {}).get('needs_accommodation', 0),
                "event_searches": signals_data.get('by_signal', {}).get('needs_events', 0)
            },
            "peak_hours": []  # TODO: Calculate from hourly stats
        }
        
    except Exception as e:
        logger.error(f"Error getting user stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/export")
async def export_stats(
    format: str = Query("json", description="Export format (json/csv)"),
    date_from: Optional[str] = Query(None, description="Start date"),
    date_to: Optional[str] = Query(None, description="End date"),
    metrics: Optional[str] = Query(None, description="Comma-separated list of metrics"),
    admin: bool = Depends(verify_admin_access),
    analytics: Any = Depends(get_analytics_manager)
):
    """
    Export statistics in various formats.
    
    Args:
        format: Export format (json/csv)
        date_from: Start date
        date_to: End date
        metrics: Specific metrics to export
    
    Returns:
        File download with exported statistics
    """
    try:
        summary = analytics.get_summary()
        
        if format == "json":
            # Return JSON file
            json_content = json.dumps(summary, indent=2)
            return StreamingResponse(
                iter([json_content]),
                media_type="application/json",
                headers={
                    "Content-Disposition": f"attachment; filename=llm_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                }
            )
        
        elif format == "csv":
            # Convert to CSV
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write headers
            writer.writerow(["Metric", "Value"])
            
            # Write basic stats
            for key, value in summary['basic_stats'].items():
                writer.writerow([key, value])
            
            # Write performance stats
            perf = summary.get('performance', {})
            for metric_type, stats in perf.items():
                if isinstance(stats, dict):
                    for stat_name, stat_value in stats.items():
                        writer.writerow([f"{metric_type}_{stat_name}", stat_value])
            
            csv_content = output.getvalue()
            
            return StreamingResponse(
                iter([csv_content]),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=llm_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                }
            )
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
        
    except Exception as e:
        logger.error(f"Error exporting stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/stats/stream")
async def stats_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time metrics streaming.
    
    Broadcasts metrics updates every 5 seconds to all connected clients.
    """
    await manager.connect(websocket)
    
    try:
        while True:
            # Keep connection alive
            # Actual metrics broadcasting will be done by a background task
            message = await websocket.receive_text()
            
            # Echo back or handle client messages
            if message == "ping":
                await websocket.send_json({"type": "pong", "timestamp": datetime.now().isoformat()})
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# Background task to broadcast metrics (to be called from main app)
async def broadcast_metrics(analytics_manager):
    """
    Background task to broadcast metrics to all connected WebSocket clients.
    Call this from your main application with a scheduler (e.g., every 5 seconds).
    
    Args:
        analytics_manager: AnalyticsManager instance
    """
    try:
        summary = analytics_manager.get_summary()
        
        # Calculate recent metrics
        total_queries = summary['basic_stats'].get('total_queries', 0)
        cache_hits = summary['basic_stats'].get('cache_hits', 0)
        cache_misses = summary['basic_stats'].get('cache_misses', 0)
        
        perf = summary.get('performance', {})
        query_latency = perf.get('query_latency', {})
        
        message = {
            "type": "metrics_update",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "queries_last_minute": 0,  # TODO: Calculate from recent data
                "avg_latency_last_minute": round(query_latency.get('avg', 0) * 1000, 2),
                "cache_hit_rate_last_minute": round(
                    cache_hits / max(cache_hits + cache_misses, 1),
                    3
                ),
                "active_queries": 0,  # TODO: Track in-flight queries
                "errors_last_minute": 0  # TODO: Calculate from recent errors
            }
        }
        
        await manager.broadcast(message)
        
    except Exception as e:
        logger.error(f"Error broadcasting metrics: {e}")


logger.info("✅ LLM Statistics Router initialized")
