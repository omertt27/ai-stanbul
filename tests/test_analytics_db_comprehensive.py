"""
Comprehensive tests for Analytics Database Service - ACTIVELY USED MODULE
Tests database operations, session tracking, and analytics functionality
"""
import pytest
import sqlite3
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
from backend.analytics_db import AnalyticsDB


class TestAnalyticsDB:
    """Test the analytics database functionality."""
    
    @pytest.fixture
    def analytics_db(self):
        """Create analytics DB instance for testing."""
        with patch('sqlite3.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_conn.cursor.return_value = mock_cursor
            mock_connect.return_value = mock_conn
            
            db = AnalyticsDB(db_path=":memory:")
            db.connection = mock_conn
            db.cursor = mock_cursor
            return db
    
    def test_initialization(self, analytics_db):
        """Test analytics database initialization."""
        assert analytics_db.db_path == ":memory:"
        assert analytics_db.connection is not None
        assert analytics_db.cursor is not None
    
    def test_create_tables(self, analytics_db):
        """Test table creation."""
        analytics_db.create_tables()
        
        # Should execute CREATE TABLE statements
        assert analytics_db.cursor.execute.called
        assert analytics_db.connection.commit.called
    
    def test_log_interaction_basic(self, analytics_db):
        """Test basic interaction logging."""
        analytics_db.log_interaction(
            session_id="test_session",
            query="Best restaurants in Istanbul",
            response="Here are some great restaurants...",
            response_time=1.5
        )
        
        # Should execute INSERT statement
        analytics_db.cursor.execute.assert_called()
        analytics_db.connection.commit.assert_called()
    
    def test_log_interaction_with_metadata(self, analytics_db):
        """Test interaction logging with metadata."""
        metadata = {
            "language": "tr",
            "location": "Sultanahmet",
            "user_agent": "Mozilla/5.0"
        }
        
        analytics_db.log_interaction(
            session_id="test_session",
            query="En iyi restoranlar",
            response="İşte harika restoranlar...",
            response_time=2.1,
            metadata=metadata
        )
        
        analytics_db.cursor.execute.assert_called()
        analytics_db.connection.commit.assert_called()
    
    def test_log_feedback(self, analytics_db):
        """Test feedback logging."""
        analytics_db.log_feedback(
            session_id="test_session",
            rating=5,
            feedback_text="Great response!",
            query="test query"
        )
        
        analytics_db.cursor.execute.assert_called()
        analytics_db.connection.commit.assert_called()
    
    def test_get_session_stats(self, analytics_db):
        """Test session statistics retrieval."""
        # Mock cursor fetchone result
        analytics_db.cursor.fetchone.return_value = (10, 4.5, 5.0)
        
        stats = analytics_db.get_session_stats("test_session")
        
        assert stats["total_interactions"] == 10
        assert stats["avg_response_time"] == 4.5
        assert stats["avg_rating"] == 5.0
        analytics_db.cursor.execute.assert_called()
    
    def test_get_popular_queries(self, analytics_db):
        """Test popular queries retrieval."""
        # Mock cursor fetchall result
        analytics_db.cursor.fetchall.return_value = [
            ("restaurants", 15),
            ("museums", 8),
            ("transportation", 5)
        ]
        
        popular = analytics_db.get_popular_queries(limit=3)
        
        assert len(popular) == 3
        assert popular[0]["query"] == "restaurants"
        assert popular[0]["count"] == 15
        analytics_db.cursor.execute.assert_called()
    
    def test_get_usage_trends(self, analytics_db):
        """Test usage trends analysis."""
        # Mock cursor fetchall result
        analytics_db.cursor.fetchall.return_value = [
            ("2024-01-01", 25),
            ("2024-01-02", 30),
            ("2024-01-03", 20)
        ]
        
        trends = analytics_db.get_usage_trends(days=7)
        
        assert len(trends) == 3
        assert trends[0]["date"] == "2024-01-01"
        assert trends[0]["interactions"] == 25
        analytics_db.cursor.execute.assert_called()
    
    def test_get_response_time_stats(self, analytics_db):
        """Test response time statistics."""
        analytics_db.cursor.fetchone.return_value = (1.2, 3.5, 2.1)
        
        stats = analytics_db.get_response_time_stats()
        
        assert stats["min_time"] == 1.2
        assert stats["max_time"] == 3.5
        assert stats["avg_time"] == 2.1
        analytics_db.cursor.execute.assert_called()
    
    def test_cleanup_old_data(self, analytics_db):
        """Test cleanup of old analytics data."""
        analytics_db.cursor.rowcount = 15
        
        result = analytics_db.cleanup_old_data(days=30)
        
        assert result["deleted_interactions"] == 15
        analytics_db.cursor.execute.assert_called()
        analytics_db.connection.commit.assert_called()
    
    def test_export_analytics_data(self, analytics_db):
        """Test analytics data export."""
        analytics_db.cursor.fetchall.return_value = [
            (1, "session1", "query1", "response1", 1.5, "2024-01-01"),
            (2, "session2", "query2", "response2", 2.0, "2024-01-02")
        ]
        
        data = analytics_db.export_analytics_data(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31)
        )
        
        assert len(data) == 2
        assert data[0]["session_id"] == "session1"
        assert data[0]["response_time"] == 1.5
        analytics_db.cursor.execute.assert_called()
    
    def test_database_connection_error(self):
        """Test database connection error handling."""
        with patch('sqlite3.connect') as mock_connect:
            mock_connect.side_effect = sqlite3.Error("Connection failed")
            
            with pytest.raises(sqlite3.Error):
                AnalyticsDB(db_path="invalid_path.db")
    
    def test_get_language_distribution(self, analytics_db):
        """Test language distribution analysis."""
        analytics_db.cursor.fetchall.return_value = [
            ("en", 45),
            ("tr", 35),
            ("ar", 20)
        ]
        
        distribution = analytics_db.get_language_distribution()
        
        assert len(distribution) == 3
        assert distribution[0]["language"] == "en"
        assert distribution[0]["count"] == 45
        analytics_db.cursor.execute.assert_called()
    
    def test_get_error_rates(self, analytics_db):
        """Test error rate analysis."""
        analytics_db.cursor.fetchall.return_value = [
            ("API_ERROR", 5),
            ("TIMEOUT", 3),
            ("INVALID_INPUT", 2)
        ]
        
        errors = analytics_db.get_error_rates()
        
        assert len(errors) == 3
        assert errors[0]["error_type"] == "API_ERROR"
        assert errors[0]["count"] == 5
        analytics_db.cursor.execute.assert_called()
    
    def test_close_connection(self, analytics_db):
        """Test database connection closing."""
        analytics_db.close()
        
        analytics_db.connection.close.assert_called()
    
    def test_context_manager(self):
        """Test analytics DB as context manager."""
        with patch('sqlite3.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn
            
            with AnalyticsDB(db_path=":memory:") as db:
                assert db.connection is not None
            
            # Should close connection when exiting context
            mock_conn.close.assert_called()
    
    def test_bulk_insert_interactions(self, analytics_db):
        """Test bulk insertion of interactions."""
        interactions = [
            ("session1", "query1", "response1", 1.5, "{}"),
            ("session2", "query2", "response2", 2.0, "{}"),
            ("session3", "query3", "response3", 1.8, "{}")
        ]
        
        analytics_db.bulk_insert_interactions(interactions)
        
        analytics_db.cursor.executemany.assert_called()
        analytics_db.connection.commit.assert_called()
    
    def test_get_peak_usage_hours(self, analytics_db):
        """Test peak usage hours analysis."""
        analytics_db.cursor.fetchall.return_value = [
            (14, 25),  # 2 PM, 25 interactions
            (15, 30),  # 3 PM, 30 interactions
            (16, 22)   # 4 PM, 22 interactions
        ]
        
        peak_hours = analytics_db.get_peak_usage_hours()
        
        assert len(peak_hours) == 3
        assert peak_hours[0]["hour"] == 14
        assert peak_hours[0]["interactions"] == 25
        analytics_db.cursor.execute.assert_called()
    
    def test_integration_full_workflow(self, analytics_db):
        """Test complete analytics workflow."""
        # 1. Create tables
        analytics_db.create_tables()
        
        # 2. Log interaction
        analytics_db.log_interaction(
            session_id="integration_test",
            query="integration test query",
            response="integration test response",
            response_time=1.5
        )
        
        # 3. Log feedback
        analytics_db.log_feedback(
            session_id="integration_test",
            rating=4,
            feedback_text="Good response"
        )
        
        # 4. Get stats
        analytics_db.cursor.fetchone.return_value = (1, 1.5, 4.0)
        stats = analytics_db.get_session_stats("integration_test")
        
        # Verify all operations were called
        assert analytics_db.cursor.execute.call_count >= 3
        assert analytics_db.connection.commit.call_count >= 2
        assert stats["total_interactions"] == 1
