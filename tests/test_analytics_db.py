"""
Comprehensive tests for Analytics Database Service
Tests user analytics, tracking, and performance metrics
"""
import pytest
import asyncio
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List
from datetime import datetime, timedelta

class TestAnalyticsDB:
    """Test Analytics Database functionality."""
    
    @pytest.fixture
    def mock_database(self):
        """Mock database for testing."""
        mock_db = MagicMock()
        mock_db.query.return_value = mock_db
        mock_db.filter.return_value = mock_db
        mock_db.group_by.return_value = mock_db
        mock_db.order_by.return_value = mock_db
        mock_db.limit.return_value = mock_db
        mock_db.all.return_value = []
        mock_db.first.return_value = None
        mock_db.count.return_value = 0
        mock_db.add.return_value = None
        mock_db.commit.return_value = None
        return mock_db
    
    @pytest.fixture
    def analytics_db(self, mock_database):
        """Create analytics database instance."""
        with patch('backend.analytics_db.get_db_session', return_value=mock_database):
            from backend.analytics_db import AnalyticsDB
            return AnalyticsDB()
    
    def test_analytics_db_initialization(self, analytics_db):
        """Test analytics database initializes correctly."""
        assert analytics_db is not None
        assert hasattr(analytics_db, 'track_user_interaction')
        assert hasattr(analytics_db, 'get_usage_statistics')
        assert hasattr(analytics_db, 'track_blog_engagement')
    
    def test_track_user_interaction_success(self, analytics_db, mock_database):
        """Test successful user interaction tracking."""
        interaction_data = {
            "session_id": "test-session-123",
            "query": "Best restaurants in Istanbul",
            "response_time": 1.5,
            "user_satisfied": True,
            "interaction_type": "query"
        }
        
        result = analytics_db.track_user_interaction(**interaction_data)
        
        assert result["status"] == "success"
        assert result["tracked"] is True
        mock_database.add.assert_called_once()
        mock_database.commit.assert_called_once()
    
    def test_track_user_interaction_with_metadata(self, analytics_db, mock_database):
        """Test user interaction tracking with additional metadata."""
        interaction_data = {
            "session_id": "metadata-session",
            "query": "Museums near Galata Tower",
            "response_time": 2.1,
            "user_satisfied": True,
            "interaction_type": "query",
            "metadata": {
                "language": "en",
                "location": "Istanbul",
                "device_type": "mobile"
            }
        }
        
        result = analytics_db.track_user_interaction(**interaction_data)
        
        assert result["status"] == "success"
        assert "metadata" in result
        assert result["metadata"]["language"] == "en"
    
    def test_get_usage_statistics_basic(self, analytics_db, mock_database):
        """Test basic usage statistics retrieval."""
        # Mock query results
        mock_database.count.return_value = 150  # Total interactions
        mock_database.all.return_value = [
            MagicMock(avg_response_time=1.8),
            MagicMock(satisfaction_rate=0.87)
        ]
        
        stats = analytics_db.get_usage_statistics(days=7)
        
        assert stats is not None
        assert "total_interactions" in stats
        assert "average_response_time" in stats
        assert "user_satisfaction_rate" in stats
        assert stats["period_days"] == 7
    
    def test_get_usage_statistics_with_filters(self, analytics_db, mock_database):
        """Test usage statistics with filtering options."""
        filters = {
            "interaction_type": "query",
            "language": "en",
            "session_type": "new_user"
        }
        
        # Mock filtered results
        mock_database.count.return_value = 75
        mock_database.all.return_value = [MagicMock(avg_response_time=1.6)]
        
        stats = analytics_db.get_usage_statistics(days=30, filters=filters)
        
        assert stats["total_interactions"] == 75
        assert stats["filters_applied"] == filters
    
    def test_track_blog_engagement_view(self, analytics_db, mock_database):
        """Test blog post view tracking."""
        engagement_data = {
            "post_id": "istanbul-guide-2025",
            "event_type": "view",
            "user_id": "user-123",
            "session_id": "blog-session"
        }
        
        result = analytics_db.track_blog_engagement(**engagement_data)
        
        assert result["status"] == "success"
        assert result["event_type"] == "view"
        assert result["post_id"] == "istanbul-guide-2025"
    
    def test_track_blog_engagement_interaction(self, analytics_db, mock_database):
        """Test blog post interaction tracking (likes, shares, comments)."""
        interactions = [
            {"post_id": "blog-1", "event_type": "like", "user_id": "user-1"},
            {"post_id": "blog-1", "event_type": "share", "user_id": "user-2"},
            {"post_id": "blog-1", "event_type": "comment", "user_id": "user-3"}
        ]
        
        results = []
        for interaction in interactions:
            result = analytics_db.track_blog_engagement(**interaction)
            results.append(result)
        
        assert all(r["status"] == "success" for r in results)
        assert len(results) == 3
        event_types = [r["event_type"] for r in results]
        assert "like" in event_types
        assert "share" in event_types
        assert "comment" in event_types
    
    def test_get_popular_queries(self, analytics_db, mock_database):
        """Test retrieval of popular user queries."""
        # Mock popular queries data
        mock_queries = [
            MagicMock(query="Best restaurants Istanbul", count=45),
            MagicMock(query="Hagia Sophia visiting hours", count=38),
            MagicMock(query="Istanbul public transport", count=32),
            MagicMock(query="Best hotels Sultanahmet", count=28),
            MagicMock(query="Turkish cuisine recommendations", count=25)
        ]
        mock_database.all.return_value = mock_queries
        
        popular_queries = analytics_db.get_popular_queries(limit=5, days=30)
        
        assert len(popular_queries) == 5
        assert popular_queries[0]["query"] == "Best restaurants Istanbul"
        assert popular_queries[0]["count"] == 45
        assert all("query" in item and "count" in item for item in popular_queries)
    
    def test_get_user_journey_analysis(self, analytics_db, mock_database):
        """Test user journey and session analysis."""
        session_id = "journey-test-session"
        
        # Mock user journey data
        mock_interactions = [
            MagicMock(query="Hotels in Istanbul", timestamp=datetime.now() - timedelta(minutes=10)),
            MagicMock(query="Best restaurants Sultanahmet", timestamp=datetime.now() - timedelta(minutes=5)),
            MagicMock(query="Hagia Sophia tickets", timestamp=datetime.now())
        ]
        mock_database.all.return_value = mock_interactions
        
        journey = analytics_db.get_user_journey_analysis(session_id)
        
        assert journey is not None
        assert "session_id" in journey
        assert "interaction_count" in journey
        assert "session_duration" in journey
        assert "query_progression" in journey
        assert journey["session_id"] == session_id
    
    def test_get_performance_metrics(self, analytics_db, mock_database):
        """Test system performance metrics retrieval."""
        # Mock performance data
        mock_database.all.return_value = [
            MagicMock(avg_response_time=1.2, p95_response_time=2.8, p99_response_time=4.1),
            MagicMock(error_rate=0.02, timeout_rate=0.005)
        ]
        
        metrics = analytics_db.get_performance_metrics(days=7)
        
        assert metrics is not None
        assert "average_response_time" in metrics
        assert "p95_response_time" in metrics
        assert "p99_response_time" in metrics
        assert "error_rate" in metrics
        assert "system_health_score" in metrics
    
    def test_get_language_distribution(self, analytics_db, mock_database):
        """Test language usage distribution analysis."""
        # Mock language data
        mock_language_data = [
            MagicMock(language="en", count=120),
            MagicMock(language="tr", count=85),
            MagicMock(language="ar", count=45),
            MagicMock(language="de", count=30),
            MagicMock(language="fr", count=20)
        ]
        mock_database.all.return_value = mock_language_data
        
        language_dist = analytics_db.get_language_distribution(days=30)
        
        assert len(language_dist) == 5
        assert language_dist[0]["language"] == "en"
        assert language_dist[0]["count"] == 120
        assert sum(item["count"] for item in language_dist) == 300
    
    def test_get_geographic_distribution(self, analytics_db, mock_database):
        """Test geographic usage distribution."""
        # Mock geographic data
        mock_geo_data = [
            MagicMock(country="Turkey", city="Istanbul", count=200),
            MagicMock(country="USA", city="New York", count=45),
            MagicMock(country="Germany", city="Berlin", count=35),
            MagicMock(country="UK", city="London", count=30)
        ]
        mock_database.all.return_value = mock_geo_data
        
        geo_dist = analytics_db.get_geographic_distribution(days=30)
        
        assert len(geo_dist) == 4
        assert geo_dist[0]["country"] == "Turkey"
        assert geo_dist[0]["city"] == "Istanbul"
        assert geo_dist[0]["count"] == 200
    
    def test_get_top_posts_by_engagement(self, analytics_db, mock_database):
        """Test top blog posts by engagement metrics."""
        # Mock blog engagement data
        mock_posts = [
            MagicMock(
                post_id="istanbul-food-guide",
                views=500,
                likes=85,
                shares=32,
                comments=18,
                engagement_score=635
            ),
            MagicMock(
                post_id="hagia-sophia-history",
                views=450,
                likes=72,
                shares=28,
                comments=15,
                engagement_score=565
            )
        ]
        mock_database.all.return_value = mock_posts
        
        top_posts = analytics_db.get_top_posts(days=7, limit=10)
        
        assert len(top_posts) == 2
        assert top_posts[0]["post_id"] == "istanbul-food-guide"
        assert top_posts[0]["engagement_score"] == 635
        assert top_posts[0]["views"] == 500
    
    def test_track_feature_usage(self, analytics_db, mock_database):
        """Test feature usage tracking."""
        feature_data = {
            "feature_name": "image_analysis",
            "session_id": "feature-test-session",
            "usage_count": 3,
            "success_rate": 0.95,
            "metadata": {
                "image_types": ["menu", "landmark"],
                "analysis_time": 2.3
            }
        }
        
        result = analytics_db.track_feature_usage(**feature_data)
        
        assert result["status"] == "success"
        assert result["feature_name"] == "image_analysis"
        assert result["tracked"] is True
    
    def test_get_conversion_funnel(self, analytics_db, mock_database):
        """Test conversion funnel analysis."""
        # Mock funnel data
        mock_funnel_data = [
            MagicMock(step="landing", count=1000),
            MagicMock(step="first_query", count=750),
            MagicMock(step="engagement", count=500),
            MagicMock(step="return_user", count=200)
        ]
        mock_database.all.return_value = mock_funnel_data
        
        funnel = analytics_db.get_conversion_funnel(days=30)
        
        assert len(funnel) == 4
        assert funnel[0]["step"] == "landing"
        assert funnel[0]["count"] == 1000
        assert funnel[0]["conversion_rate"] == 1.0  # First step is 100%
        assert funnel[1]["conversion_rate"] == 0.75  # 750/1000
    
    def test_real_time_analytics(self, analytics_db, mock_database):
        """Test real-time analytics dashboard data."""
        # Mock real-time data
        current_time = datetime.now()
        mock_real_time_data = {
            "active_sessions": 45,
            "queries_last_hour": 128,
            "average_response_time_last_hour": 1.4,
            "error_rate_last_hour": 0.01,
            "top_queries_last_hour": [
                "restaurants near me",
                "museum hours",
                "weather Istanbul"
            ]
        }
        
        with patch.object(analytics_db, '_get_real_time_metrics', return_value=mock_real_time_data):
            real_time_data = analytics_db.get_real_time_dashboard()
        
        assert real_time_data["active_sessions"] == 45
        assert real_time_data["queries_last_hour"] == 128
        assert real_time_data["average_response_time_last_hour"] == 1.4
        assert len(real_time_data["top_queries_last_hour"]) == 3
    
    def test_error_handling_database_failure(self, analytics_db, mock_database):
        """Test error handling when database operations fail."""
        mock_database.add.side_effect = Exception("Database connection failed")
        
        interaction_data = {
            "session_id": "error-session",
            "query": "Test query",
            "response_time": 1.0,
            "user_satisfied": True,
            "interaction_type": "query"
        }
        
        result = analytics_db.track_user_interaction(**interaction_data)
        
        assert result["status"] == "error"
        assert "database" in result["message"].lower()
    
    def test_data_retention_cleanup(self, analytics_db, mock_database):
        """Test automatic data retention and cleanup."""
        # Mock old data
        retention_days = 365
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        mock_database.filter.return_value.count.return_value = 50  # 50 old records
        
        cleanup_result = analytics_db.cleanup_old_data(retention_days=retention_days)
        
        assert cleanup_result["status"] == "success"
        assert cleanup_result["deleted_records"] == 50
        assert cleanup_result["retention_days"] == retention_days
    
    def test_analytics_export(self, analytics_db, mock_database):
        """Test analytics data export functionality."""
        export_params = {
            "date_range": {"start": "2025-09-01", "end": "2025-09-21"},
            "data_types": ["interactions", "blog_engagement", "performance"],
            "format": "json"
        }
        
        # Mock export data
        mock_export_data = {
            "interactions": [{"query": "test", "response_time": 1.0}],
            "blog_engagement": [{"post_id": "post-1", "views": 100}],
            "performance": [{"metric": "response_time", "value": 1.2}]
        }
        
        with patch.object(analytics_db, '_generate_export_data', return_value=mock_export_data):
            export_result = analytics_db.export_analytics_data(**export_params)
        
        assert export_result["status"] == "success"
        assert export_result["format"] == "json"
        assert "data" in export_result
        assert len(export_result["data"]["interactions"]) == 1
    
    @pytest.mark.asyncio
    async def test_async_analytics_operations(self, analytics_db, mock_database):
        """Test asynchronous analytics operations for performance."""
        # Test concurrent tracking operations
        interactions = [
            {"session_id": f"async-session-{i}", "query": f"Query {i}", "response_time": 1.0}
            for i in range(10)
        ]
        
        # Mock async tracking
        async def mock_track_async(interaction):
            return {"status": "success", "tracked": True}
        
        with patch.object(analytics_db, 'track_user_interaction_async', side_effect=mock_track_async):
            tasks = [
                analytics_db.track_user_interaction_async(**interaction)
                for interaction in interactions
            ]
            
            results = await asyncio.gather(*tasks)
        
        assert len(results) == 10
        assert all(result["status"] == "success" for result in results)
