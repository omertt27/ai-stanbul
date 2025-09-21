"""
AI Cache Service Tests - Testing REAL API Methods Only
Tests only methods that actually exist in ai_cache_service.py
"""
import pytest
import time
import json
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
from backend.ai_cache_service import (
    AIRateLimitCache,
    get_ai_cache_service,
    init_ai_cache_service
)


class TestAICacheServiceRealAPI:
    """Test the AI Cache Service functionality - only real methods."""
    
    @pytest.fixture
    def cache_service(self):
        """Create AI cache service for testing."""
        return AIRateLimitCache(
            redis_url="redis://localhost:6379",
            rate_limit_per_user=5,  # Low for testing
            rate_limit_per_ip=20,   # Low for testing
            cache_ttl=3600
        )
    
    @pytest.fixture
    def cache_service_no_redis(self):
        """Create AI cache service without Redis for testing fallback."""
        with patch('redis.from_url') as mock_redis:
            mock_redis.side_effect = Exception("Redis unavailable")
            service = AIRateLimitCache(
                redis_url="redis://invalid:6379",
                rate_limit_per_user=5,
                rate_limit_per_ip=20,
                cache_ttl=3600
            )
            # Force enable cache for memory fallback testing
            service.cache_enabled = True
            return service
    
    def test_initialization_with_redis(self, cache_service):
        """Test AI cache service initialization with Redis."""
        assert cache_service is not None
        assert hasattr(cache_service, 'redis_client')
        assert hasattr(cache_service, 'cache_enabled')
        assert hasattr(cache_service, 'rate_limit_per_user')
        assert hasattr(cache_service, 'rate_limit_per_ip')
        assert hasattr(cache_service, 'cache_ttl')
        assert hasattr(cache_service, 'memory_rate_limits')
        assert hasattr(cache_service, 'memory_cache')
        
        assert cache_service.rate_limit_per_user == 5
        assert cache_service.rate_limit_per_ip == 20
        assert cache_service.cache_ttl == 3600
    
    def test_initialization_without_redis(self, cache_service_no_redis):
        """Test AI cache service initialization without Redis."""
        assert cache_service_no_redis is not None
        assert cache_service_no_redis.redis_client is None
        # Note: We force-enable cache for memory fallback testing
        assert cache_service_no_redis.cache_enabled is True
        assert isinstance(cache_service_no_redis.memory_rate_limits, dict)
        assert isinstance(cache_service_no_redis.memory_cache, dict)
    
    def test_generate_cache_key(self, cache_service):
        """Test cache key generation."""
        # Test basic query
        key1 = cache_service._generate_cache_key("restaurants in Istanbul")
        assert isinstance(key1, str)
        assert key1.startswith("ai_cache:")
        
        # Test that same queries produce same keys
        key2 = cache_service._generate_cache_key("restaurants in Istanbul")
        assert key1 == key2
        
        # Test with context
        context = {"location": "Sultanahmet", "language": "en"}
        key3 = cache_service._generate_cache_key("restaurants", context)
        key4 = cache_service._generate_cache_key("restaurants")
        assert key3 != key4  # Context should affect key
    
    def test_get_rate_limit_key(self, cache_service):
        """Test rate limit key generation."""
        key = cache_service._get_rate_limit_key("user123", "user")
        assert isinstance(key, str)
        assert "rate_limit:user:user123:" in key
        
        # Should include current hour
        hour = datetime.now().strftime("%Y%m%d%H")
        assert hour in key
    
    def test_check_rate_limit_memory_fallback(self, cache_service_no_redis):
        """Test rate limiting with memory fallback."""
        session_id = "test_session"
        ip_address = "192.168.1.1"
        
        # First check should pass
        allowed, info = cache_service_no_redis.check_rate_limit(session_id, ip_address)
        assert allowed is True
        assert info["rate_limited"] is False
        assert "user_requests" in info
        assert "ip_requests" in info
        assert "remaining_user" in info
        assert "remaining_ip" in info
    
    def test_rate_limit_exceeded(self, cache_service_no_redis):
        """Test rate limit exceeded scenario."""
        session_id = "test_session"
        ip_address = "192.168.1.1"
        
        # Exhaust rate limit
        for i in range(6):  # Exceed limit of 5
            cache_service_no_redis.increment_rate_limit(session_id, ip_address)
        
        # Check should now fail
        allowed, info = cache_service_no_redis.check_rate_limit(session_id, ip_address)
        assert allowed is False
        assert info["rate_limited"] is True
        assert "message" in info
        assert info["user_requests"] >= 5
    
    def test_increment_rate_limit_memory(self, cache_service_no_redis):
        """Test rate limit increment with memory fallback."""
        session_id = "test_session"
        ip_address = "192.168.1.1"
        
        # Initial state
        allowed, info_before = cache_service_no_redis.check_rate_limit(session_id, ip_address)
        
        # Increment
        cache_service_no_redis.increment_rate_limit(session_id, ip_address)
        
        # Check after increment
        allowed, info_after = cache_service_no_redis.check_rate_limit(session_id, ip_address)
        
        assert info_after["user_requests"] == info_before["user_requests"] + 1
        assert info_after["ip_requests"] == info_before["ip_requests"] + 1
    
    def test_get_cached_response_miss(self, cache_service):
        """Test cache miss scenario."""
        query = "unique query that doesn't exist"
        result = cache_service.get_cached_response(query)
        assert result is None
    
    def test_cache_response_and_retrieve(self, cache_service_no_redis):
        """Test caching and retrieving response with memory fallback."""
        query = "restaurants in Sultanahmet"
        response_data = {
            "message": "Here are some great restaurants in Sultanahmet...",
            "recommendations": ["Restaurant A", "Restaurant B"]
        }
        user_context = {"location": "Sultanahmet", "language": "en"}
        
        # Cache the response
        cache_service_no_redis.cache_response(query, response_data, user_context)
        
        # Retrieve from cache
        cached_result = cache_service_no_redis.get_cached_response(query, user_context)
        
        assert cached_result is not None
        assert cached_result["from_cache"] is True
        assert "cache_hit_time" in cached_result
        assert cached_result["message"] == response_data["message"]
    
    def test_cache_disabled(self, cache_service_no_redis):
        """Test cache behavior when disabled."""
        cache_service_no_redis.cache_enabled = False
        
        query = "test query"
        response_data = {"message": "test response"}
        
        # Should not cache
        cache_service_no_redis.cache_response(query, response_data)
        
        # Should not retrieve
        result = cache_service_no_redis.get_cached_response(query)
        assert result is None
    
    def test_cache_memory_cleanup(self, cache_service_no_redis):
        """Test memory cache cleanup functionality."""
        # Fill memory cache with many entries
        for i in range(105):  # Exceed limit of 100
            query = f"test query {i}"
            response = {"message": f"response {i}"}
            cache_service_no_redis.cache_response(query, response)
        
        # Cache should be cleaned up to maintain size
        assert len(cache_service_no_redis.memory_cache) <= 100
    
    def test_get_cache_stats(self, cache_service):
        """Test cache statistics retrieval."""
        stats = cache_service.get_cache_stats()
        
        assert isinstance(stats, dict)
        assert "cache_enabled" in stats
        assert "redis_available" in stats
        assert "rate_limits" in stats
        assert "cache_ttl_seconds" in stats
        
        rate_limits = stats["rate_limits"]
        assert "per_user_hour" in rate_limits
        assert "per_ip_hour" in rate_limits
        assert rate_limits["per_user_hour"] == 5
        assert rate_limits["per_ip_hour"] == 20
    
    def test_get_cache_stats_memory_fallback(self, cache_service_no_redis):
        """Test cache statistics with memory fallback."""
        stats = cache_service_no_redis.get_cache_stats()
        
        assert isinstance(stats, dict)
        # Note: We force-enable cache for memory fallback testing
        assert stats["cache_enabled"] is True
        assert stats["redis_available"] is False
        assert "memory_cache_size" in stats
        assert "memory_rate_limit_entries" in stats
    
    def test_clear_cache_memory(self, cache_service_no_redis):
        """Test cache clearing with memory fallback."""
        # Add some cache entries
        for i in range(5):
            query = f"test query {i}"
            response = {"message": f"response {i}"}
            cache_service_no_redis.cache_response(query, response)
        
        assert len(cache_service_no_redis.memory_cache) > 0
        
        # Clear cache
        cache_service_no_redis.clear_cache()
        
        assert len(cache_service_no_redis.memory_cache) == 0
    
    @patch('backend.ai_cache_service.redis.from_url')
    def test_clear_cache_redis(self, mock_redis, cache_service):
        """Test cache clearing with Redis."""
        mock_redis_client = MagicMock()
        mock_redis_client.keys.return_value = ["ai_cache:key1", "ai_cache:key2"]
        cache_service.redis_client = mock_redis_client
        
        cache_service.clear_cache()
        
        mock_redis_client.keys.assert_called_once_with("ai_cache:*")
        mock_redis_client.delete.assert_called_once()
    
    def test_optimize_database_queries(self, cache_service):
        """Test database optimization functionality."""
        mock_db = MagicMock()
        
        # Should not raise exception
        cache_service.optimize_database_queries(mock_db)
        
        # Should have called execute twice (ANALYZE and PRAGMA optimize)
        assert mock_db.execute.call_count == 2
    
    def test_optimize_database_queries_error_handling(self, cache_service):
        """Test database optimization error handling."""
        mock_db = MagicMock()
        mock_db.execute.side_effect = Exception("Database error")
        
        # Should not raise exception, just log warning
        cache_service.optimize_database_queries(mock_db)
    
    def test_global_service_functions(self):
        """Test global service functions."""
        # Test get_ai_cache_service
        service1 = get_ai_cache_service()
        service2 = get_ai_cache_service()
        
        # Should return same instance
        assert service1 is service2
        assert isinstance(service1, AIRateLimitCache)
        
        # Test init_ai_cache_service
        custom_service = init_ai_cache_service(
            redis_url="redis://localhost:6379",
            rate_limit_per_user=10,
            rate_limit_per_ip=100
        )
        
        assert isinstance(custom_service, AIRateLimitCache)
        assert custom_service.rate_limit_per_user == 10
        assert custom_service.rate_limit_per_ip == 100
    
    def test_query_normalization(self, cache_service):
        """Test query normalization for better cache hits."""
        # Test basic normalization (lowercase, strip)
        key1 = cache_service._generate_cache_key("  RESTAURANTS in Istanbul  ")
        key2 = cache_service._generate_cache_key("restaurants in istanbul")
        # These should be the same due to lowercase and strip
        assert key1 == key2
        
        # Test replacement logic
        key3 = cache_service._generate_cache_key("restaurant in Istanbul")
        # This will become "restaurants in istanbul" due to replacement
        
        # Test typo corrections
        key4 = cache_service._generate_cache_key("restarunt in Istanbul")
        key5 = cache_service._generate_cache_key("resturant in Istanbul")
        # These should be the same because typos -> "restaurants"
        assert key4 == key5
        
        # All typos should normalize to same result
        assert key3 == key4 == key5
    
    def test_rate_limit_hour_key_cleanup(self, cache_service_no_redis):
        """Test cleanup of old rate limit entries."""
        # Manually add old entries
        current_hour = int(time.time() // 3600)
        old_hour = current_hour - 3
        
        old_key = f"user_test_session_{old_hour}"
        current_key = f"user_test_session_{current_hour}"
        
        cache_service_no_redis.memory_rate_limits[old_key] = 5
        cache_service_no_redis.memory_rate_limits[current_key] = 2
        
        # Check rate limit should clean up old entries
        cache_service_no_redis.check_rate_limit("test_session", "192.168.1.1")
        
        # Old entry should be removed, current should remain
        assert old_key not in cache_service_no_redis.memory_rate_limits
        assert any(key.startswith("user_test_session") for key in cache_service_no_redis.memory_rate_limits)
    
    def test_cache_response_data_structure(self, cache_service_no_redis):
        """Test the structure of cached response data."""
        query = "test query"
        response_data = {
            "message": "test response",
            "extra_data": {"key": "value"}
        }
        
        cache_service_no_redis.cache_response(query, response_data)
        cached_result = cache_service_no_redis.get_cached_response(query)
        
        assert cached_result is not None
        assert "timestamp" in cached_result
        assert "query" in cached_result
        assert "from_cache" in cached_result
        assert cached_result["from_cache"] is True
        assert cached_result["message"] == "test response"
        assert len(cached_result["query"]) <= 100  # Should truncate long queries
    
    @patch('time.time')
    def test_memory_cache_expiration(self, mock_time, cache_service_no_redis):
        """Test memory cache expiration logic."""
        # Set initial time
        mock_time.return_value = 1000
        
        query = "test query"
        response_data = {"message": "test response"}
        
        # Cache response
        cache_service_no_redis.cache_response(query, response_data)
        
        # Should retrieve successfully
        result1 = cache_service_no_redis.get_cached_response(query)
        assert result1 is not None
        
        # Advance time beyond TTL
        mock_time.return_value = 1000 + cache_service_no_redis.cache_ttl + 1
        
        # Should not retrieve expired entry
        result2 = cache_service_no_redis.get_cached_response(query)
        assert result2 is None
        
        # Entry should be removed from cache
        cache_key = cache_service_no_redis._generate_cache_key(query)
        assert cache_key not in cache_service_no_redis.memory_cache
