"""
Admin API Endpoints for Semantic Cache Monitoring

Add these endpoints to your FastAPI backend to expose semantic cache metrics.

Usage:
    1. Add this code to your main FastAPI app (e.g., backend/main.py)
    2. Or create a separate router and include it
    3. Access at: /api/admin/semantic-cache/stats

Example integration in main.py:
    from admin_semantic_cache_api import router as semantic_cache_router
    app.include_router(semantic_cache_router, prefix="/api/admin", tags=["admin"])
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import redis
import json
import time
from datetime import datetime

# Create router
router = APIRouter()


class SemanticCacheStats(BaseModel):
    """Semantic cache statistics model."""
    timestamp: str
    total_entries: int
    cache_hit_rate: float
    keyspace_hits: int
    keyspace_misses: int
    memory_used: str
    performance_status: str
    recommendations: List[str]


class SemanticCacheSample(BaseModel):
    """Sample cache entry model."""
    query: str
    response_preview: str
    timestamp: float
    age_hours: float
    similarity_used: Optional[float] = None


class SemanticCacheDetailedStats(BaseModel):
    """Detailed semantic cache statistics."""
    timestamp: str
    total_entries: int
    cache_hit_rate: float
    keyspace_hits: int
    keyspace_misses: int
    memory_used: str
    memory_used_bytes: int
    performance_status: str
    recommendations: List[str]
    sample_entries: List[SemanticCacheSample]
    cache_size_limit: int
    cache_ttl_hours: float


# Redis connection (adjust based on your config)
def get_redis_client():
    """Get Redis client (adjust connection details as needed)."""
    try:
        return redis.Redis(
            host='localhost',  # Use env var in production
            port=6379,
            db=0,
            decode_responses=True
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Redis connection failed: {str(e)}")


@router.get("/semantic-cache/stats", response_model=SemanticCacheStats)
async def get_semantic_cache_stats():
    """
    Get semantic cache statistics.
    
    Returns:
        - Total cache entries
        - Cache hit rate
        - Memory usage
        - Performance status
        - Recommendations
    """
    try:
        redis_client = get_redis_client()
        
        # Get cache keys
        cache_prefix = "semantic_cache:"
        keys = redis_client.keys(f"{cache_prefix}*")
        total_entries = len(keys)
        
        # Get Redis stats
        redis_info = redis_client.info('stats')
        keyspace_hits = redis_info.get('keyspace_hits', 0)
        keyspace_misses = redis_info.get('keyspace_misses', 0)
        
        # Calculate hit rate
        total_ops = keyspace_hits + keyspace_misses
        hit_rate = (keyspace_hits / total_ops * 100) if total_ops > 0 else 0
        
        # Get memory usage
        memory_info = redis_client.info('memory')
        used_memory = memory_info.get('used_memory_human', 'N/A')
        
        # Determine performance status
        if hit_rate >= 40:
            status = "EXCELLENT"
        elif hit_rate >= 30:
            status = "GOOD"
        elif hit_rate >= 20:
            status = "FAIR"
        else:
            status = "LOW"
        
        # Generate recommendations
        recommendations = []
        if hit_rate < 20:
            recommendations.extend([
                "Consider decreasing similarity threshold (current: 0.85)",
                "Increase top-K candidates for better matching",
                "Analyze query diversity - queries may be too varied"
            ])
        elif hit_rate > 70:
            recommendations.extend([
                "Verify response quality is maintained",
                "Check if queries are too repetitive"
            ])
        else:
            recommendations.append("Performance is within expected range")
        
        if total_entries > 8000:
            recommendations.append("Cache approaching size limit - consider increasing max size")
        
        return SemanticCacheStats(
            timestamp=datetime.now().isoformat(),
            total_entries=total_entries,
            cache_hit_rate=round(hit_rate, 2),
            keyspace_hits=keyspace_hits,
            keyspace_misses=keyspace_misses,
            memory_used=used_memory,
            performance_status=status,
            recommendations=recommendations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")


@router.get("/semantic-cache/detailed", response_model=SemanticCacheDetailedStats)
async def get_semantic_cache_detailed():
    """
    Get detailed semantic cache statistics with sample entries.
    
    Returns:
        - Complete cache statistics
        - Sample cache entries
        - Configuration details
    """
    try:
        redis_client = get_redis_client()
        
        # Get basic stats (reuse logic)
        cache_prefix = "semantic_cache:"
        keys = redis_client.keys(f"{cache_prefix}*")
        total_entries = len(keys)
        
        redis_info = redis_client.info('stats')
        keyspace_hits = redis_info.get('keyspace_hits', 0)
        keyspace_misses = redis_info.get('keyspace_misses', 0)
        total_ops = keyspace_hits + keyspace_misses
        hit_rate = (keyspace_hits / total_ops * 100) if total_ops > 0 else 0
        
        memory_info = redis_client.info('memory')
        used_memory = memory_info.get('used_memory_human', 'N/A')
        used_memory_bytes = memory_info.get('used_memory', 0)
        
        # Performance status
        if hit_rate >= 40:
            status = "EXCELLENT"
        elif hit_rate >= 30:
            status = "GOOD"
        elif hit_rate >= 20:
            status = "FAIR"
        else:
            status = "LOW"
        
        # Recommendations
        recommendations = []
        if hit_rate < 20:
            recommendations.extend([
                "Consider decreasing similarity threshold",
                "Increase top-K candidates",
                "Analyze query diversity"
            ])
        elif hit_rate > 70:
            recommendations.extend([
                "Verify response quality",
                "Check query repetition"
            ])
        else:
            recommendations.append("Performance optimal")
        
        # Sample entries
        sample_keys = keys[:10] if len(keys) > 0 else []
        sample_entries = []
        
        for key in sample_keys:
            try:
                data = json.loads(redis_client.get(key))
                age_hours = (time.time() - data.get('timestamp', time.time())) / 3600
                
                sample_entries.append(SemanticCacheSample(
                    query=data.get('query', '')[:100],
                    response_preview=data.get('response', '')[:100],
                    timestamp=data.get('timestamp', 0),
                    age_hours=round(age_hours, 2),
                    similarity_used=data.get('similarity', None)
                ))
            except:
                continue
        
        # Config from env (you may want to read from actual config)
        cache_size_limit = 10000  # Default
        cache_ttl_hours = 24  # Default
        
        return SemanticCacheDetailedStats(
            timestamp=datetime.now().isoformat(),
            total_entries=total_entries,
            cache_hit_rate=round(hit_rate, 2),
            keyspace_hits=keyspace_hits,
            keyspace_misses=keyspace_misses,
            memory_used=used_memory,
            memory_used_bytes=used_memory_bytes,
            performance_status=status,
            recommendations=recommendations,
            sample_entries=sample_entries,
            cache_size_limit=cache_size_limit,
            cache_ttl_hours=cache_ttl_hours
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get detailed stats: {str(e)}")


@router.post("/semantic-cache/clear")
async def clear_semantic_cache():
    """
    Clear all semantic cache entries.
    
    ⚠️ USE WITH CAUTION: This will remove all cached responses.
    """
    try:
        redis_client = get_redis_client()
        
        # Get all semantic cache keys
        cache_prefix = "semantic_cache:"
        keys = redis_client.keys(f"{cache_prefix}*")
        
        if not keys:
            return {
                "status": "success",
                "message": "No cache entries to clear",
                "deleted_count": 0
            }
        
        # Delete all keys
        deleted = redis_client.delete(*keys)
        
        return {
            "status": "success",
            "message": f"Cleared {deleted} cache entries",
            "deleted_count": deleted
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@router.get("/semantic-cache/health")
async def semantic_cache_health():
    """
    Check semantic cache health and configuration.
    
    Returns:
        - Redis connection status
        - Cache configuration
        - Current status
    """
    try:
        redis_client = get_redis_client()
        
        # Ping Redis
        redis_client.ping()
        
        # Get basic stats
        keys = redis_client.keys("semantic_cache:*")
        total_entries = len(keys)
        
        return {
            "status": "healthy",
            "redis_connected": True,
            "total_entries": total_entries,
            "timestamp": datetime.now().isoformat()
        }
        
    except redis.ConnectionError:
        return {
            "status": "unhealthy",
            "redis_connected": False,
            "error": "Redis connection failed",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


# Optional: Add authentication/authorization
# from fastapi import Security
# from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
#
# security = HTTPBearer()
#
# async def verify_admin(credentials: HTTPAuthorizationCredentials = Security(security)):
#     """Verify admin token."""
#     # Implement your auth logic here
#     pass
#
# Then add to endpoints:
# @router.get("/semantic-cache/stats", dependencies=[Depends(verify_admin)])
