"""
Analytics DB Tests - Testing REAL API Methods Only
Tests only methods that actually exist in analytics_db.py
"""
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
from backend.analytics_db import AnalyticsDB


class TestAnalyticsDBRealAPI:
    """Test the Analytics DB functionality - only real methods."""
    
    @pytest.fixture
    def mock_sqlite_connect(self):
        """Mock sqlite3.connect for testing."""
        with patch('backend.analytics_db.sqlite3.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_conn.execute.return_value = mock_cursor
            mock_conn.__enter__.return_value = mock_conn
            mock_connect.return_value = mock_conn
            yield mock_conn, mock_cursor
    
    @pytest.fixture
    def analytics_db(self, mock_sqlite_connect):
        """Create analytics DB instance for testing."""
        return AnalyticsDB(db_path=":memory:")
    
    def test_initialization(self, analytics_db):
        """Test analytics DB initialization."""
        assert analytics_db.db_path == ":memory:"
        assert hasattr(analytics_db, 'lock')
        assert analytics_db.lock is not None
    
    def test_track_page_view_success(self, analytics_db, mock_sqlite_connect):
        """Test successful page view tracking."""
        mock_conn, mock_cursor = mock_sqlite_connect
        
        page_path = "/blog/istanbul-guide"
        user_agent = "Mozilla/5.0"
        ip_address = "127.0.0.1"
        session_id = "test_session_123"
        
        # Should not raise exception
        analytics_db.track_page_view(page_path, user_agent, ip_address, session_id)
        
        # Verify database was called
        assert mock_conn.execute.called
        assert mock_conn.commit.called
    
    def test_track_page_view_minimal(self, analytics_db, mock_sqlite_connect):
        """Test page view tracking with minimal parameters."""
        mock_conn, mock_cursor = mock_sqlite_connect
        
        page_path = "/home"
        
        # Should work with just page_path
        analytics_db.track_page_view(page_path)
        
        assert mock_conn.execute.called
    
    def test_track_blog_engagement_success(self, analytics_db, mock_sqlite_connect):
        """Test successful blog engagement tracking."""
        mock_conn, mock_cursor = mock_sqlite_connect
        
        post_id = "istanbul-travel-guide"
        event_type = "like"
        user_session = "test_session_123"
        metadata = {"source": "homepage"}
        
        analytics_db.track_blog_engagement(post_id, event_type, user_session, metadata)
        
        assert mock_conn.execute.called
        assert mock_conn.commit.called
    
    def test_update_active_session_success(self, analytics_db, mock_sqlite_connect):
        """Test successful active session update."""
        mock_conn, mock_cursor = mock_sqlite_connect
        
        session_id = "test_session_123"
        current_page = "/blog/istanbul-guide"
        
        analytics_db.update_active_session(session_id, current_page)
        
        assert mock_conn.execute.called
        assert mock_conn.commit.called
    
    def test_get_active_readers_count_success(self, analytics_db, mock_sqlite_connect):
        """Test successful active readers count retrieval."""
        mock_conn, mock_cursor = mock_sqlite_connect
        
        # Mock cursor return value
        mock_cursor.fetchone.return_value = [5]
        
        count = analytics_db.get_active_readers_count(minutes=5)
        
        assert count == 5
        assert mock_cursor.fetchone.called
    
    def test_get_todays_stats_success(self, analytics_db, mock_sqlite_connect):
        """Test successful today's stats retrieval."""
        mock_conn, mock_cursor = mock_sqlite_connect
        
        # Mock multiple cursor returns for different queries
        mock_cursor.fetchone.side_effect = [
            [100],  # page_views_today
            [25],   # blog_reads_today
        ]
        mock_cursor.fetchall.return_value = [
            ('like', 5),
            ('share', 3),
            ('comment', 2)
        ]
        
        stats = analytics_db.get_todays_stats()
        
        assert stats['page_views_today'] == 100
        assert stats['blog_reads_today'] == 25
        assert stats['likes_today'] == 5
        assert stats['shares_today'] == 3
        assert stats['comments_today'] == 2
        assert 'engagement_today' in stats
    
    def test_get_top_posts_success(self, analytics_db, mock_sqlite_connect):
        """Test successful top posts retrieval."""
        mock_conn, mock_cursor = mock_sqlite_connect
        
        # Mock cursor return value
        mock_cursor.fetchall.return_value = [
            ('post1', 100, 10, 5, 115),
            ('post2', 80, 8, 3, 91),
            ('post3', 60, 6, 2, 68)
        ]
        
        posts = analytics_db.get_top_posts(days=7, limit=5)
        
        assert len(posts) == 3
        assert posts[0]['post_id'] == 'post1'
        assert posts[0]['views'] == 100
        assert posts[0]['likes'] == 10
        assert posts[0]['shares'] == 5
        assert 'engagement_rate' in posts[0]
        assert 'total_engagement' in posts[0]
    
    def test_get_hourly_engagement_rate_success(self, analytics_db, mock_sqlite_connect):
        """Test successful hourly engagement rate retrieval."""
        mock_conn, mock_cursor = mock_sqlite_connect
        
        # Mock cursor return value
        mock_cursor.fetchall.return_value = [
            ('like', 10),
            ('share', 5),
            ('comment', 3),
            ('view', 50)
        ]
        
        engagement = analytics_db.get_hourly_engagement_rate()
        
        assert engagement['likes_per_hour'] == 10
        assert engagement['shares_per_hour'] == 5
        assert engagement['comments_per_hour'] == 3
        assert engagement['views_per_hour'] == 50
    
    def test_cleanup_old_sessions_success(self, analytics_db, mock_sqlite_connect):
        """Test successful old sessions cleanup."""
        mock_conn, mock_cursor = mock_sqlite_connect
        
        # Should not raise exception
        analytics_db.cleanup_old_sessions(hours=24)
        
        assert mock_conn.execute.called
        assert mock_conn.commit.called
