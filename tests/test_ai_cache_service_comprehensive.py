"""
Comprehensive tests for AI Cache Service - ACTIVELY USED MODULE
Tests all major functionality: caching, rate limiting, cache management
"""
import pytest
import time
from unittest.mock import patch, MagicMock
from backend.ai_cache_service import AIRateLimitCache


class TestAIRateLimitCache:
    """Test the AI cache service functionality."""
    
    @pytest.fixture
    def cache_service(self):
        """Create cache service instance for testing."""
        with patch('redis.from_url') as mock_redis:
            # Mock Redis connection that works
            mock_redis_instance = MagicMock()
            mock_redis_instance.ping.return_value = True
            mock_redis_instance.get.return_value = None
            mock_redis_instance.set.return_value = True
            mock_redis_instance.incr.return_value = 1
            mock_redis_instance.setex.return_value = True
            mock_redis_instance.expire.return_value = True
            mock_redis.return_value = mock_redis_instance
            
            service = AIRateLimitCache(
                redis_url="redis://localhost:6379",
                rate_limit_per_user=20,
                rate_limit_per_ip=100,
                cache_ttl=3600
            )
            # Store mock for test access
            setattr(service, 'mock_redis', mock_redis_instance)
            return service
    
    @pytest.fixture
    def cache_service_no_redis(self):
        """Create cache service without Redis for fallback testing."""
        with patch('redis.from_url') as mock_redis:
            # Mock Redis connection that fails
            mock_redis.side_effect = Exception("Redis connection failed")
            
            service = AIRateLimitCache()
            return service
    
    def test_initialization_with_redis(self, cache_service):
        """Test successful initialization with Redis."""
        assert cache_service.cache_enabled is True
        assert cache_service.rate_limit_per_user == 20
        assert cache_service.rate_limit_per_ip == 100
        assert cache_service.cache_ttl == 3600
    
    def test_initialization_without_redis(self, cache_service_no_redis):
        """Test fallback initialization without Redis."""
        assert cache_service_no_redis.cache_enabled is False
        assert cache_service_no_redis.redis_client is None
        assert isinstance(cache_service_no_redis.memory_rate_limits, dict)
        assert isinstance(cache_service_no_redis.memory_cache, dict)
    
    def test_generate_cache_key_basic(self, cache_service):
        """Test cache key generation with basic query."""
        key1 = cache_service._generate_cache_key("Best restaurants in Istanbul")
        key2 = cache_service._generate_cache_key("best restaurants in istanbul")
        
        # Should generate same key for case variations
        assert key1 == key2
        assert key1.startswith("ai_cache:")
        assert len(key1) > 20  # MD5 hash should be present
    
    def test_generate_cache_key_with_context(self, cache_service):
        """Test cache key generation with user context."""
        context = {"location": "Sultanahmet", "language": "tr"}
        key = cache_service._generate_cache_key("restaurants", context)
        
        assert key.startswith("ai_cache:")
        
        # Different context should generate different key
        context2 = {"location": "Beyoglu", "language": "en"}
        key2 = cache_service._generate_cache_key("restaurants", context2)
        assert key != key2
    
    def test_generate_cache_key_normalization(self, cache_service):
        """Test query normalization in cache key generation."""
        # Test common typo corrections
        key1 = cache_service._generate_cache_key("restaurant in Istanbul")
        key2 = cache_service._generate_cache_key("restaurants in Istanbul")
        key3 = cache_service._generate_cache_key("restarunt in Istanbul")
        key4 = cache_service._generate_cache_key("resturant in Istanbul")
        
        # All should normalize to same key
        assert key1 == key2 == key3 == key4
    
    def test_rate_limit_key_generation(self, cache_service):
        """Test rate limit key generation."""
        key = cache_service._get_rate_limit_key("test_session", "user")
        assert key.startswith("rate_limit:user:test_session:")
        assert len(key.split(":")) == 4  # Should have 4 parts
    
    def test_rate_limit_check_allowed(self, cache_service):
        """Test rate limit check when requests are allowed."""
        # Mock Redis returns low counts
        mock_redis = getattr(cache_service, 'mock_redis')
        mock_redis.get.side_effect = ["5", "10"]  # user_count, ip_count
        
        allowed, info = cache_service.check_rate_limit("test_session", "127.0.0.1")
        
        assert allowed is True
        assert info["rate_limited"] is False
        assert info["user_requests"] == 5
        assert info["ip_requests"] == 10
        assert info["remaining_user"] == 15  # 20 - 5
        assert info["remaining_ip"] == 90    # 100 - 10
    
    def test_rate_limit_check_user_exceeded(self, cache_service):
        """Test rate limit check when user limit is exceeded."""
        # Mock Redis returns high user count
        mock_redis = getattr(cache_service, 'mock_redis')
        mock_redis.get.side_effect = ["25", "10"]  # user_count > limit
        
        allowed, info = cache_service.check_rate_limit("test_session", "127.0.0.1")
        
        assert allowed is False
        assert info["rate_limited"] is True
        assert info["user_requests"] == 25
        assert "Rate limit exceeded" in info["message"]
        assert "reset_time" in info
    
    def test_rate_limit_check_ip_exceeded(self, cache_service):
        """Test rate limit check when IP limit is exceeded."""
        # Mock Redis returns high IP count
        mock_redis = getattr(cache_service, 'mock_redis')
        mock_redis.get.side_effect = ["5", "105"]  # ip_count > limit
        
        allowed, info = cache_service.check_rate_limit("test_session", "127.0.0.1")
        
        assert allowed is False
        assert info["rate_limited"] is True
        assert info["ip_requests"] == 105
    
    def test_rate_limit_fallback_to_memory(self, cache_service):
        """Test rate limit falls back to memory when Redis fails."""
        # Make Redis raise exception
        mock_redis = getattr(cache_service, 'mock_redis')
        mock_redis.get.side_effect = Exception("Redis error")
        
        # Should still work with memory fallback
        allowed, info = cache_service.check_rate_limit("test_session", "127.0.0.1")
        
        assert allowed is True  # Should allow with fresh memory cache
        assert info["rate_limited"] is False
    
    def test_memory_based_rate_limiting(self, cache_service_no_redis):
        """Test pure memory-based rate limiting when Redis is unavailable."""
        # First request should be allowed
        allowed1, info1 = cache_service_no_redis.check_rate_limit("test_session", "127.0.0.1")
        assert allowed1 is True
        assert info1["rate_limited"] is False
        
        # Simulate many requests by directly manipulating memory
        import time
        hour_key = int(time.time() // 3600)
        user_key = f"user_test_session_{hour_key}"
        cache_service_no_redis.memory_rate_limits[user_key] = 25  # Exceed limit
        
        allowed2, info2 = cache_service_no_redis.check_rate_limit("test_session", "127.0.0.1")
        assert allowed2 is False
        assert info2["rate_limited"] is True
    
    def test_get_cached_response_basic(self, cache_service):
        """Test basic cache retrieval."""
        # Mock Redis returns cached data
        cached_data = '{"message": "Great restaurants in Istanbul", "from_cache": false}'
        cache_service.mock_redis.get.return_value = cached_data
        
        result = cache_service.get_cached_response("Best restaurants")
        
        assert result is not None
        assert result["message"] == "Great restaurants in Istanbul"
        assert result["from_cache"] is True  # Should be set by get_cached_response
    
    def test_get_cached_response_miss(self, cache_service):
        """Test cache miss (no cached data)."""
        # Mock Redis returns None
        cache_service.mock_redis.get.return_value = None
        
        result = cache_service.get_cached_response("New query")
        assert result is None
    
    def test_get_cached_response_invalid_json(self, cache_service):
        """Test cache with invalid JSON data."""
        # Mock Redis returns invalid JSON
        cache_service.mock_redis.get.return_value = "invalid json data"
        
        result = cache_service.get_cached_response("test query")
        assert result is None
    
    def test_cache_response_basic(self, cache_service):
        """Test basic cache setting."""
        response_data = {"message": "Istanbul is beautiful", "timestamp": "2024-01-01"}
        
        cache_service.cache_response("Istanbul tourism", response_data)
        
        # Verify Redis setex was called (cache_response uses setex, not set+expire)
        cache_service.mock_redis.setex.assert_called()
    
    def test_cache_response_with_context(self, cache_service):
        """Test cache setting with user context."""
        response_data = {"message": "Turkish restaurants"}
        context = {"location": "Beyoglu", "language": "tr"}
        
        cache_service.cache_response("restaurants", response_data, context)
        
        # Should still call Redis setex
        cache_service.mock_redis.setex.assert_called()
    
    def test_increment_rate_limit(self, cache_service):
        """Test incrementing rate limit counters."""
        cache_service.increment_rate_limit("test_session", "127.0.0.1")
        
        # Should call Redis incr and expire
        assert cache_service.mock_redis.incr.call_count == 2  # user + ip
        assert cache_service.mock_redis.expire.call_count == 2
    
    def test_increment_rate_limit_memory_fallback(self, cache_service):
        """Test incrementing rate limit with Redis failure."""
        # Make Redis fail
        cache_service.mock_redis.incr.side_effect = Exception("Redis error")
        
        # Should still work (memory fallback)
        cache_service.increment_rate_limit("test_session", "127.0.0.1")
        
        # Should have created memory entries
        assert len(cache_service.memory_rate_limits) > 0
    
    def test_clear_cache(self, cache_service):
        """Test cache clearing functionality."""
        cache_service.mock_redis.keys.return_value = ["ai_cache:key1", "ai_cache:key2"]
        cache_service.mock_redis.delete.return_value = 2
        
        cache_service.clear_cache()
        
        # Should call keys and delete
        cache_service.mock_redis.keys.assert_called_with("ai_cache:*")
        cache_service.mock_redis.delete.assert_called()
    
    def test_get_cache_stats(self, cache_service):
        """Test cache statistics retrieval."""
        # Mock Redis info response
        cache_service.mock_redis.info.return_value = {
            "used_memory_human": "1.5M",
            "connected_clients": 5
        }
        
        stats = cache_service.get_cache_stats()
        
        assert "cache_enabled" in stats
        assert "redis_available" in stats
        assert "rate_limits" in stats
        assert stats["redis_info"]["memory_usage"] == "1.5M"
        assert stats["redis_info"]["connected_clients"] == 5
    
    def test_get_cache_stats_redis_error(self, cache_service):
        """Test cache stats when Redis fails."""
        cache_service.mock_redis.info.side_effect = Exception("Redis error")
        
        stats = cache_service.get_cache_stats()
        
        assert stats["redis_info"]["connected"] is False
        assert stats["cache_enabled"] is True  # Cache service itself is enabled
    
    def test_cleanup_old_entries_memory(self, cache_service_no_redis):
        """Test cleanup of old memory entries."""
        import time
        current_hour = int(time.time() // 3600)
        old_hour = current_hour - 3
        
        # Add old and new entries
        cache_service_no_redis.memory_rate_limits[f"user_test_{old_hour}"] = 5
        cache_service_no_redis.memory_rate_limits[f"user_test_{current_hour}"] = 3
        cache_service_no_redis.memory_cache[f"cache_{old_hour}"] = {"data": "old"}
        cache_service_no_redis.memory_cache[f"cache_{current_hour}"] = {"data": "new"}
        
        # Trigger cleanup by checking rate limit
        cache_service_no_redis.check_rate_limit("test_session", "127.0.0.1")
        
        # Old entries should be cleaned up
        old_keys = [k for k in cache_service_no_redis.memory_rate_limits.keys() if str(old_hour) in k]
        assert len(old_keys) == 0
    
    def test_integration_full_flow(self, cache_service):
        """Test complete flow: rate limit check, cache get/set, record request."""
        # 1. Check rate limit (should be allowed)
        allowed, info = cache_service.check_rate_limit("integration_test", "192.168.1.1")
        assert allowed is True
        
        # 2. Check cache (should be miss)
        cache_service.mock_redis.get.return_value = None
        cached = cache_service.get_cached_response("integration test query")
        assert cached is None
        
        # 3. Set cache with response
        response = {"message": "Integration test response", "confidence": 0.9}
        cache_service.cache_response("integration test query", response)
        
        # 4. Increment rate limit
        cache_service.increment_rate_limit("integration_test", "192.168.1.1")
        
        # Verify all Redis operations were called
        assert cache_service.mock_redis.setex.called
        assert cache_service.mock_redis.incr.called
        assert cache_service.mock_redis.expire.called
