"""
Focused tests for actively used modules - improving coverage
Tests actual methods that exist in the backend modules
"""
import pytest
import asyncio
from unittest.mock import patch, MagicMock
from backend.ai_cache_service import AIRateLimitCache
from backend.gdpr_service import GDPRService


class TestAIRateLimitCacheFocused:
    """Test AI cache service with actual methods."""
    
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
            mock_redis_instance.expire.return_value = True
            mock_redis.return_value = mock_redis_instance
            
            service = AIRateLimitCache()
            return service
    
    def test_cache_key_generation(self, cache_service):
        """Test cache key generation method."""
        key = cache_service._generate_cache_key("test query")
        assert key.startswith("ai_cache:")
        assert len(key) > 20
    
    def test_rate_limit_key_generation(self, cache_service):
        """Test rate limit key generation."""
        key = cache_service._get_rate_limit_key("session_123", "user")
        assert "rate_limit:user:session_123:" in key
    
    def test_check_rate_limit_allowed(self, cache_service):
        """Test rate limit checking when allowed."""
        allowed, info = cache_service.check_rate_limit("test_session", "127.0.0.1")
        assert isinstance(allowed, bool)
        assert isinstance(info, dict)
    
    def test_increment_rate_limit(self, cache_service):
        """Test rate limit increment."""
        # Should not raise exception
        cache_service.increment_rate_limit("test_session", "127.0.0.1")
    
    def test_get_cached_response_miss(self, cache_service):
        """Test cache retrieval when no cache exists."""
        result = cache_service.get_cached_response("new query")
        assert result is None
    
    def test_cache_response(self, cache_service):
        """Test caching a response."""
        response = {"answer": "Test response"}
        cache_service.cache_response("test query", response)
        # Should not raise exception
    
    def test_get_cache_stats(self, cache_service):
        """Test cache statistics retrieval."""
        stats = cache_service.get_cache_stats()
        assert isinstance(stats, dict)
    
    def test_clear_cache(self, cache_service):
        """Test cache clearing."""
        result = cache_service.clear_cache()
        # May return None or dict, should not raise exception


class TestGDPRServiceFocused:
    """Test GDPR service with actual methods."""
    
    @pytest.fixture
    def gdpr_service(self):
        """Create GDPR service instance."""
        return GDPRService()
    
    @pytest.fixture
    def mock_db_session(self):
        """Mock database session."""
        with patch('backend.gdpr_service.SessionLocal') as mock_session_local:
            mock_session = MagicMock()
            mock_session_local.return_value = mock_session
            yield mock_session
    
    def test_initialization(self, gdpr_service):
        """Test GDPR service initialization."""
        assert gdpr_service.retention_periods is not None
        assert gdpr_service.personal_data_categories is not None
    
    def test_hash_identifier(self, gdpr_service):
        """Test identifier hashing."""
        hash1 = gdpr_service._hash_identifier("test_session_123")
        hash2 = gdpr_service._hash_identifier("test_session_123")
        assert hash1 == hash2  # Same input, same hash
        assert len(hash1) == 16  # Truncated to 16 chars
    
    def test_create_audit_log(self, gdpr_service, mock_db_session):
        """Test audit log creation."""
        gdpr_service.create_audit_log(
            "test_action", 
            "test_subject", 
            {"test": "details"}
        )
        # Should not raise exception
        assert mock_db_session.close.called


class TestAnalyticsDBFocused:
    """Test analytics database with actual methods."""
    
    def test_import_analytics_db(self):
        """Test importing analytics database module."""
        from backend.analytics_db import AnalyticsDB
        assert AnalyticsDB is not None


class TestMultimodalAIFocused:
    """Test multimodal AI with actual methods."""
    
    def test_import_multimodal_ai(self):
        """Test importing multimodal AI module."""
        from backend.api_clients.multimodal_ai import MultimodalAIService
        assert MultimodalAIService is not None
    
    def test_multimodal_service_creation(self):
        """Test creating multimodal AI service."""
        from backend.api_clients.multimodal_ai import MultimodalAIService
        service = MultimodalAIService()
        assert service is not None


class TestRealTimeDataFocused:
    """Test real-time data with actual methods."""
    
    def test_import_realtime_data(self):
        """Test importing realtime data module."""
        from backend.api_clients.realtime_data import RealTimeDataAggregator
        assert RealTimeDataAggregator is not None
    
    def test_aggregator_creation(self):
        """Test creating data aggregator."""
        from backend.api_clients.realtime_data import RealTimeDataAggregator
        aggregator = RealTimeDataAggregator()
        assert aggregator is not None
        assert hasattr(aggregator, 'event_client')
        assert hasattr(aggregator, 'crowd_client')
        assert hasattr(aggregator, 'traffic_client')


# Integration test - test a realistic workflow
class TestIntegrationWorkflow:
    """Test realistic integration workflows."""
    
    def test_cache_workflow(self):
        """Test cache service workflow."""
        with patch('redis.from_url') as mock_redis:
            mock_redis_instance = MagicMock()
            mock_redis_instance.ping.return_value = True
            mock_redis.return_value = mock_redis_instance
            
            # Create service
            cache_service = AIRateLimitCache()
            
            # Check rate limit
            allowed, info = cache_service.check_rate_limit("session_123", "127.0.0.1")
            assert isinstance(allowed, bool)
            
            # Try to get cached response
            cached = cache_service.get_cached_response("test query")
            # May be None (cache miss)
            
            # Cache a response
            response = {"answer": "Test answer"}
            cache_service.cache_response("test query", response)
            
            # Get stats
            stats = cache_service.get_cache_stats()
            assert isinstance(stats, dict)
    
    def test_gdpr_workflow(self):
        """Test GDPR service workflow."""
        with patch('backend.gdpr_service.SessionLocal') as mock_session_local:
            mock_session = MagicMock()
            mock_session_local.return_value = mock_session
            
            # Create service
            gdpr_service = GDPRService()
            
            # Create audit log
            gdpr_service.create_audit_log(
                "data_access",
                "session_123", 
                {"request_type": "chat_history"}
            )
            
            # Hash an identifier
            hashed = gdpr_service._hash_identifier("sensitive_data")
            assert len(hashed) == 16
