# Redis Caching Implementation Guide

## Overview
Redis caching has been added to AI Istanbul backend to improve response times and reduce database load.

## Features
✅ **Automatic Redis Integration**: Falls back to in-memory cache if Redis unavailable
✅ **Easy-to-use Decorators**: Simple caching for API endpoints  
✅ **Flexible TTL**: Configurable time-to-live for different data types
✅ **Pattern-based Invalidation**: Clear cache by pattern
✅ **Statistics**: Built-in cache hit/miss tracking

## Setup

### 1. Render Redis Setup (Production)
1. Go to Render Dashboard
2. Click "New +" → "Redis"
3. Name: `ai-istanbul-redis`
4. Plan: Free (25MB, perfect for caching)
5. Click "Create Redis"
6. Copy the **Internal Redis URL**
7. Go to your web service → Environment
8. Add: `REDIS_URL=<your-internal-redis-url>`
9. Save (auto-redeploys)

### 2. Local Development
```bash
# Install Redis locally (macOS)
brew install redis
brew services start redis

# Or use Docker
docker run -d -p 6379:6379 redis:latest

# Set environment variable
export REDIS_URL=redis://localhost:6379
```

## Usage

### Method 1: Using the Caching Decorator

```python
from core.caching import cache_response

@router.get("/restaurants")
@cache_response(ttl=1800, key_prefix="restaurants")
async def get_restaurants(location: str):
    # This will be cached for 30 minutes
    return {"restaurants": [...]}
```

### Method 2: Manual Caching

```python
from services.redis_cache import get_cache_service

cache = get_cache_service()

# Cache a value
await cache.set("my_key", {"data": "value"}, ttl=3600)

# Get cached value
result = await cache.get("my_key")

# Delete
await cache.delete("my_key")

# Clear pattern
await cache.clear_pattern("restaurants:*")
```

### Method 3: Specialized Methods

```python
from services.redis_cache import get_cache_service

cache = get_cache_service()

# Cache chat response
await cache.cache_chat_response(
    message="restaurants in Beyoğlu",
    language="en",
    response="Here are 5 restaurants...",
    ttl=3600
)

# Get cached chat response
response = await cache.get_cached_chat_response(
    message="restaurants in Beyoğlu",
    language="en"
)

# Cache restaurant query
await cache.cache_restaurant_query(
    query_params={"location": "Beyoğlu", "cuisine": "Turkish"},
    results=[...],
    ttl=1800
)
```

## TTL Recommendations

- **Chat Responses**: 3600s (1 hour) - user queries repeat often
- **Restaurant Queries**: 1800s (30 min) - semi-static data
- **Museum/Attraction Info**: 3600s (1 hour) - rarely changes
- **Events**: 3600s (1 hour) - updated daily
- **User Preferences**: 300s (5 min) - may change frequently

## Cache Statistics

```python
from services.redis_cache import get_cache_service

cache = get_cache_service()
stats = await cache.get_stats()

# Returns:
# {
#     "enabled": True,
#     "type": "redis",  # or "memory"
#     "hits": 1234,
#     "misses": 567,
#     "keys": 890
# }
```

## Invalidation Strategies

### 1. Manual Invalidation
```python
# When restaurant data is updated
await cache.invalidate_data_cache("restaurants")

# When events are added
await cache.invalidate_data_cache("events")
```

### 2. Automatic TTL
Most cache entries expire automatically based on TTL

### 3. Pattern-based
```python
# Clear all chat caches
await cache.clear_pattern("chat:*")

# Clear specific location caches
await cache.clear_pattern("restaurants:Beyoğlu:*")
```

## Monitoring

Check cache status via health endpoint:
```bash
curl https://ai-stanbul.onrender.com/api/health/detailed
```

Look for `cache_service` section in response.

## Performance Benefits

### Expected Improvements:
- 🚀 **Response Time**: 50-90% faster for cached requests
- 📊 **Database Load**: 60-80% reduction
- 💰 **Cost Savings**: Lower database usage = lower costs
- 👥 **User Experience**: Near-instant responses for repeat queries

### Example:
- **Without Cache**: 800ms (DB query + LLM)
- **With Cache**: 50ms (Redis lookup)
- **Improvement**: 16x faster! ⚡

## Troubleshooting

### Redis Connection Issues
If Redis is unavailable, the system automatically falls back to in-memory caching:
```
⚠️ Redis connection failed: ... Using in-memory cache.
```

This is expected and normal - the app continues working.

### Clear All Cache
```bash
# Via Render Shell
redis-cli FLUSHDB

# Or programmatically
from services.redis_cache import get_cache_service
cache = get_cache_service()
await cache.clear_pattern("*")
```

### Check Redis Connection
```bash
# Via Render Shell
redis-cli PING
# Should return: PONG
```

## Next Steps

1. ✅ Deploy with Redis environment variable
2. ✅ Monitor cache hit rate in health endpoint
3. ✅ Adjust TTL values based on usage patterns
4. ✅ Add caching decorators to more endpoints

## Files Changed

- `backend/services/redis_cache.py` - Main cache service
- `backend/core/caching.py` - Caching decorators
- `backend/core/startup.py` - Redis initialization
- `backend/main_modular.py` - Shutdown handler
- `backend/requirements.txt` - Redis async dependency
- `REDIS_CACHING_GUIDE.md` - This file

## Production Checklist

- [ ] Add REDIS_URL to Render environment
- [ ] Deploy latest code
- [ ] Verify Redis connection in logs
- [ ] Check cache stats in /api/health/detailed
- [ ] Monitor response times
- [ ] Adjust TTL values if needed

---

**Status**: ✅ Ready for Production  
**Next**: Add REDIS_URL environment variable on Render and deploy
