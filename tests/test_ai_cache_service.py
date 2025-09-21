"""
Comprehensive tests for AI Cache Service
Tests caching, rate limiting, and performance optimization features
"""
import pytest
import asyncio
import time
from unittest.mock import patch, MagicMock
from typing import Dict, Any

class TestAICacheService:
    """Test AI Cache Service functionality."""
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client for testing."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        mock_redis.set.return_value = True
        mock_redis.delete.return_value = True
        mock_redis.exists.return_value = False
        mock_redis.incr.return_value = 1
        mock_redis.expire.return_value = True
        mock_redis.keys.return_value = []
        return mock_redis
    
    @pytest.fixture
    def cache_service(self, mock_redis):
        """Create cache service instance for testing."""
        with patch('ai_cache_service.redis.Redis', return_value=mock_redis):
            from ai_cache_service import AICacheService
            service = AICacheService(redis_url="redis://localhost:6379")
            return service
    
    def test_cache_service_initialization(self, mock_redis):
        """Test cache service initializes correctly."""
        with patch('ai_cache_service.redis.Redis', return_value=mock_redis):
            from ai_cache_service import AICacheService
            service = AICacheService(redis_url="redis://localhost:6379")
            assert service.redis_client is not None
            assert service.cache_ttl == 3600  # Default TTL
    
    def test_generate_cache_key(self, cache_service):
        """Test cache key generation."""
        query = "Best restaurants in Istanbul"
        context = {"location": "Sultanahmet", "language": "en"}
        
        key = cache_service._generate_cache_key(query, context)
        assert isinstance(key, str)
        assert len(key) > 0
        assert "restaurants" in key.lower() or "cache" in key
    
    def test_cache_response_success(self, cache_service, mock_redis):
        """Test successful response caching."""
        query = "Best museums in Istanbul"
        response = {"message": "Great museums to visit include Hagia Sophia..."}
        context = {"language": "en"}
        
        mock_redis.set.return_value = True
        
        result = cache_service.cache_response(query, response, context)
        assert result is True
        mock_redis.set.assert_called_once()
    
    def test_get_cached_response_hit(self, cache_service, mock_redis):
        """Test cache hit scenario."""
        query = "Best restaurants"
        context = {"language": "en"}
        cached_data = '{"message": "Great restaurants include..."}'
        
        mock_redis.get.return_value = cached_data.encode('utf-8')
        
        result = cache_service.get_cached_response(query, context)
        assert result is not None
        assert "message" in result
        mock_redis.get.assert_called_once()
    
    def test_get_cached_response_miss(self, cache_service, mock_redis):
        """Test cache miss scenario."""
        query = "New query not in cache"
        context = {"language": "en"}
        
        mock_redis.get.return_value = None
        
        result = cache_service.get_cached_response(query, context)
        assert result is None
        mock_redis.get.assert_called_once()
    
    def test_rate_limiting_check_allowed(self, cache_service, mock_redis):
        """Test rate limiting when user is within limits."""
        session_id = "test-session-123"
        client_ip = "192.168.1.1"
        
        mock_redis.get.return_value = b'5'  # 5 requests made
        
        allowed, info = cache_service.check_rate_limit(session_id, client_ip)
        assert allowed is True
        assert isinstance(info, dict)
        assert "requests_made" in info
        assert "limit" in info
    
    def test_rate_limiting_check_exceeded(self, cache_service, mock_redis):
        """Test rate limiting when user exceeds limits."""
        session_id = "test-session-456"
        client_ip = "192.168.1.2"
        
        mock_redis.get.return_value = b'100'  # 100 requests made (over limit)
        
        allowed, info = cache_service.check_rate_limit(session_id, client_ip)
        assert allowed is False
        assert isinstance(info, dict)
        assert info["requests_made"] >= info["limit"]
    
    def test_increment_rate_limit(self, cache_service, mock_redis):
        """Test rate limit increment."""
        session_id = "test-session-789"
        client_ip = "192.168.1.3"
        
        mock_redis.incr.return_value = 6
        
        cache_service.increment_rate_limit(session_id, client_ip)
        mock_redis.incr.assert_called()
        mock_redis.expire.assert_called()
    
    def test_clear_cache(self, cache_service, mock_redis):
        """Test cache clearing functionality."""
        mock_redis.keys.return_value = [b'cache:key1', b'cache:key2']
        mock_redis.delete.return_value = 2
        
        result = cache_service.clear_cache()
        assert result is True
        mock_redis.keys.assert_called()
        mock_redis.delete.assert_called()
    
    def test_get_cache_stats(self, cache_service, mock_redis):
        """Test cache statistics retrieval."""
        mock_redis.get.side_effect = [b'100', b'80', b'1000']  # hits, misses, total_requests
        
        stats = cache_service.get_cache_stats()
        assert isinstance(stats, dict)
        assert "hit_rate" in stats
        assert "total_requests" in stats
        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert 0 <= stats["hit_rate"] <= 1
    
    def test_cache_invalidation(self, cache_service, mock_redis):
        """Test cache invalidation for specific patterns."""
        pattern = "restaurants"
        mock_redis.keys.return_value = [b'cache:restaurants:1', b'cache:restaurants:2']
        mock_redis.delete.return_value = 2
        
        result = cache_service.invalidate_cache_pattern(pattern)
        assert result is True
        mock_redis.keys.assert_called()
    
    def test_error_handling_redis_connection(self, mock_redis):
        """Test error handling when Redis is unavailable."""
        mock_redis.get.side_effect = Exception("Redis connection failed")
        
        with patch('ai_cache_service.redis.Redis', return_value=mock_redis):
            from ai_cache_service import AICacheService
            service = AICacheService(redis_url="redis://localhost:6379")
            
            # Should handle errors gracefully
            result = service.get_cached_response("test query", {})
            assert result is None
    
    def test_cache_key_consistency(self, cache_service):
        """Test that identical inputs produce identical cache keys."""
        query = "Best Turkish restaurants"
        context = {"location": "Istanbul", "language": "en"}
        
        key1 = cache_service._generate_cache_key(query, context)
        key2 = cache_service._generate_cache_key(query, context)
        
        assert key1 == key2
    
    def test_cache_key_uniqueness(self, cache_service):
        """Test that different inputs produce different cache keys."""
        query1 = "Best restaurants"
        query2 = "Best museums"
        context = {"language": "en"}
        
        key1 = cache_service._generate_cache_key(query1, context)
        key2 = cache_service._generate_cache_key(query2, context)
        
        assert key1 != key2
    
    def test_ttl_configuration(self, cache_service, mock_redis):
        """Test TTL (Time To Live) configuration for cached items."""
        query = "Test query"
        response = {"message": "Test response"}
        context = {}
        
        cache_service.cache_response(query, response, context)
        
        # Verify TTL was set
        args, kwargs = mock_redis.set.call_args
        assert 'ex' in kwargs or len(args) >= 3  # TTL parameter present
    
    @pytest.mark.asyncio
    async def test_async_cache_operations(self, cache_service, mock_redis):
        """Test cache operations work correctly in async context."""
        query = "Async test query"
        response = {"message": "Async response"}
        context = {"async": True}
        
        # Test caching in async context
        cache_service.cache_response(query, response, context)
        
        # Small delay to simulate async operations
        await asyncio.sleep(0.01)
        
        # Test retrieval
        mock_redis.get.return_value = b'{"message": "Async response"}'
        result = cache_service.get_cached_response(query, context)
        
        assert result is not None
        assert result["message"] == "Async response"
    
    def test_memory_usage_optimization(self, cache_service, mock_redis):
        """Test cache doesn't store excessively large responses."""
        query = "Large response test"
        large_response = {"message": "x" * 10000}  # 10KB response
        context = {}
        
        # Should handle large responses appropriately
        result = cache_service.cache_response(query, large_response, context)
        
        # Verify it was processed (either cached or rejected based on size limits)
        assert isinstance(result, bool)
    
    def test_cache_performance_metrics(self, cache_service, mock_redis):
        """Test performance metrics tracking."""
        # Simulate cache hits and misses
        mock_redis.get.side_effect = [None, b'{"cached": "data"}', None]
        
        # Cache miss
        cache_service.get_cached_response("query1", {})
        
        # Cache hit  
        cache_service.get_cached_response("query2", {})
        
        # Cache miss
        cache_service.get_cached_response("query3", {})
        
        # Get stats should show performance metrics
        mock_redis.get.side_effect = [b'1', b'2', b'3']  # hits, misses, total
        stats = cache_service.get_cache_stats()
        
        assert "hit_rate" in stats
        assert isinstance(stats["hit_rate"], (int, float))
