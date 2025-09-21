"""
GDPR Service Tests - Testing REAL API Methods Only
Tests only methods that actually exist in gdpr_service.py
"""
import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from backend.gdpr_service import GDPRService


class TestGDPRServiceRealAPI:
    """Test the GDPR service functionality - only real methods."""
    
    @pytest.fixture
    def gdpr_service(self):
        """Create GDPR service instance for testing."""
        return GDPRService()
    
    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        with patch('backend.gdpr_service.SessionLocal') as mock_session_local:
            mock_session = MagicMock()
            mock_session_local.return_value = mock_session
            yield mock_session
    
    def test_initialization(self, gdpr_service):
        """Test GDPR service initialization."""
        assert gdpr_service.retention_periods is not None
        assert gdpr_service.personal_data_categories is not None
        assert isinstance(gdpr_service.retention_periods, dict)
        assert isinstance(gdpr_service.personal_data_categories, dict)
    
    def test_retention_periods_configuration(self, gdpr_service):
        """Test retention periods are properly configured."""
        periods = gdpr_service.retention_periods
        assert 'chat_sessions' in periods
        assert 'user_feedback' in periods
        assert 'analytics_data' in periods
        assert 'consent_records' in periods
        assert 'audit_logs' in periods
        
        # Check reasonable values
        assert periods['chat_sessions'] == 30
        assert periods['consent_records'] == 2555  # 7 years
    
    def test_personal_data_categories(self, gdpr_service):
        """Test personal data categories are defined."""
        categories = gdpr_service.personal_data_categories
        assert 'session_data' in categories
        assert 'technical_data' in categories
        assert 'usage_data' in categories
        assert 'feedback_data' in categories
        assert 'consent_data' in categories
    
    def test_hash_identifier(self, gdpr_service):
        """Test identifier hashing for privacy."""
        identifier = "test_session_123"
        hashed = gdpr_service._hash_identifier(identifier)
        
        assert hashed != identifier
        assert len(hashed) == 16  # Truncated SHA256
        assert isinstance(hashed, str)
        
        # Same input should produce same hash
        hashed2 = gdpr_service._hash_identifier(identifier)
        assert hashed == hashed2
    
    def test_create_audit_log_success(self, gdpr_service, mock_db_session):
        """Test successful audit log creation."""
        action = "data_access_request"
        data_subject = "test_session_123"
        details = {"email": "test@example.com"}
        
        # Mock successful database operations
        mock_db_session.execute.return_value = None
        mock_db_session.commit.return_value = None
        
        # Should not raise exception
        gdpr_service.create_audit_log(action, data_subject, details)
        
        # Verify database was called
        assert mock_db_session.execute.called
        assert mock_db_session.commit.called
    
    def test_create_audit_log_database_error(self, gdpr_service, mock_db_session):
        """Test audit log creation with database error."""
        action = "data_access_request"
        data_subject = "test_session_123"
        details = {"email": "test@example.com"}
        
        # Mock database error
        mock_db_session.execute.side_effect = Exception("Database error")
        
        # Should handle exception gracefully
        gdpr_service.create_audit_log(action, data_subject, details)
        
        # Verify database was called despite error
        assert mock_db_session.execute.called
    
    def test_handle_data_access_request_success(self, gdpr_service, mock_db_session):
        """Test successful data access request handling."""
        session_id = "test_session_123"
        email = "test@example.com"
        
        # Mock successful database operations
        mock_db_session.execute.return_value.fetchall.return_value = []
        mock_db_session.execute.return_value.fetchone.return_value = None
        
        result = gdpr_service.handle_data_access_request(session_id, email)
        
        assert result['status'] == 'success'
        assert 'message' in result
        assert 'data' in result
        assert result['data']['request_info']['session_id'] == session_id
    
    def test_handle_data_access_request_error(self, gdpr_service, mock_db_session):
        """Test data access request with database error."""
        session_id = "test_session_123"
        email = "test@example.com"
        
        # Mock database error that causes the main method to fail
        def mock_execute_side_effect(*args, **kwargs):
            raise Exception("Database error")
        
        mock_db_session.execute.side_effect = mock_execute_side_effect
        
        result = gdpr_service.handle_data_access_request(session_id, email)
        
        # The method should still return success but with empty data
        # because _collect_user_data handles its own errors
        assert result['status'] == 'success'
        assert 'data' in result
        assert result['data']['user_data'] == {}  # Empty due to error
    
    def test_handle_data_deletion_request_success(self, gdpr_service, mock_db_session):
        """Test successful data deletion request handling."""
        session_id = "test_session_123"
        email = "test@example.com"
        
        # Mock successful database operations
        mock_result = MagicMock()
        mock_result.rowcount = 5
        mock_db_session.execute.return_value = mock_result
        
        result = gdpr_service.handle_data_deletion_request(session_id, email)
        
        assert result['status'] == 'success'
        assert 'message' in result
        assert 'deletion_summary' in result
    
    def test_handle_data_deletion_request_error(self, gdpr_service, mock_db_session):
        """Test data deletion request with database error."""
        session_id = "test_session_123"
        email = "test@example.com"
        
        # Mock database error
        mock_db_session.execute.side_effect = Exception("Database error")
        
        result = gdpr_service.handle_data_deletion_request(session_id, email)
        
        assert result['status'] == 'error'
        assert 'message' in result
        assert 'Failed to process' in result['message']
    
    def test_collect_user_data_success(self, gdpr_service, mock_db_session):
        """Test successful user data collection."""
        session_id = "test_session_123"
        
        # Mock database responses
        mock_db_session.execute.return_value.fetchall.side_effect = [
            [('2023-01-01', 'Hello', 'Hi there', 'en')],  # chat_history
            [('2023-01-01', 'positive', 5, 'Great!')],     # feedback_data
            [('analytics', True, '2023-01-01', '1.0')]      # consent_data
        ]
        mock_db_session.execute.return_value.fetchone.return_value = (
            '2023-01-01', '2023-01-01', '127.0.0.1', 'Test Agent'
        )
        
        user_data = gdpr_service._collect_user_data(session_id)
        
        assert 'chat_history' in user_data
        assert 'feedback' in user_data
        assert 'session_info' in user_data
        assert 'consent_records' in user_data
    
    def test_collect_user_data_error(self, gdpr_service, mock_db_session):
        """Test user data collection with database error."""
        session_id = "test_session_123"
        
        # Mock database error
        mock_db_session.execute.side_effect = Exception("Database error")
        
        user_data = gdpr_service._collect_user_data(session_id)
        
        # Should return empty dict on error
        assert user_data == {}
    
    def test_delete_user_data_success(self, gdpr_service, mock_db_session):
        """Test successful user data deletion."""
        session_id = "test_session_123"
        
        # Mock successful database operations
        mock_result = MagicMock()
        mock_result.rowcount = 3
        mock_db_session.execute.return_value = mock_result
        
        deletion_summary = gdpr_service._delete_user_data(session_id)
        
        assert 'session_id' in deletion_summary
        assert 'deletion_timestamp' in deletion_summary
        assert 'deleted_records' in deletion_summary
        assert deletion_summary['session_id'] == session_id
    
    def test_record_consent_success(self, gdpr_service, mock_db_session):
        """Test successful consent recording."""
        session_id = "test_session_123"
        consent_data = {
            'analytics': True,
            'marketing': False,
            'version': '1.0'
        }
        
        # Mock successful database operations
        mock_db_session.execute.return_value = None
        mock_db_session.commit.return_value = None
        
        # Should not raise exception
        gdpr_service.record_consent(session_id, consent_data)
        
        # Verify database was called
        assert mock_db_session.execute.called
        assert mock_db_session.commit.called
    
    def test_record_consent_error(self, gdpr_service, mock_db_session):
        """Test consent recording with database error."""
        session_id = "test_session_123"
        consent_data = {'analytics': True}
        
        # Mock database error
        mock_db_session.execute.side_effect = Exception("Database error")
        
        # Should handle exception gracefully
        gdpr_service.record_consent(session_id, consent_data)
    
    def test_cleanup_expired_data_success(self, gdpr_service, mock_db_session):
        """Test successful expired data cleanup."""
        # Mock successful database operations
        mock_result = MagicMock()
        mock_result.rowcount = 10
        mock_db_session.execute.return_value = mock_result
        
        cleanup_summary = gdpr_service.cleanup_expired_data()
        
        assert isinstance(cleanup_summary, dict)
        # Should not include consent_records or audit_logs
        assert 'consent_records' not in cleanup_summary
        assert 'audit_logs' not in cleanup_summary
    
    def test_cleanup_expired_data_error(self, gdpr_service, mock_db_session):
        """Test expired data cleanup with database error."""
        # Mock database error
        mock_db_session.execute.side_effect = Exception("Database error")
        
        cleanup_summary = gdpr_service.cleanup_expired_data()
        
        # Should return empty dict on error
        assert cleanup_summary == {}
    
    def test_get_consent_status_success(self, gdpr_service, mock_db_session):
        """Test successful consent status retrieval."""
        session_id = "test_session_123"
        
        # Mock database response
        mock_db_session.execute.return_value.fetchall.return_value = [
            ('analytics', True, '2023-01-01'),
            ('marketing', False, '2023-01-01')
        ]
        
        consent_status = gdpr_service.get_consent_status(session_id)
        
        assert 'analytics' in consent_status
        assert 'marketing' in consent_status
        assert consent_status['analytics']['granted'] is True
        assert consent_status['marketing']['granted'] is False
    
    def test_get_consent_status_error(self, gdpr_service, mock_db_session):
        """Test consent status retrieval with database error."""
        session_id = "test_session_123"
        
        # Mock database error
        mock_db_session.execute.side_effect = Exception("Database error")
        
        consent_status = gdpr_service.get_consent_status(session_id)
        
        # Should return empty dict on error
        assert consent_status == {}
    
    def test_send_data_export_email(self, gdpr_service):
        """Test data export email notification."""
        email = "test@example.com"
        export_data = {"user_data": {"chat_history": []}}
        
        # Should not raise exception (just logs for now)
        gdpr_service._send_data_export_email(email, export_data)
    
    def test_send_deletion_confirmation_email(self, gdpr_service):
        """Test deletion confirmation email notification."""
        email = "test@example.com"
        deletion_summary = {"deleted_records": {"chat_history": 5}}
        
        # Should not raise exception (just logs for now)
        gdpr_service._send_deletion_confirmation_email(email, deletion_summary)
    
    def test_integration_full_workflow(self, gdpr_service, mock_db_session):
        """Test complete GDPR workflow with real methods."""
        session_id = "test_session_123"
        email = "test@example.com"
        
        # Mock all database operations
        mock_db_session.execute.return_value.fetchall.return_value = []
        mock_db_session.execute.return_value.fetchone.return_value = None
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_db_session.execute.return_value = mock_result
        
        # Step 1: Record consent
        consent_data = {'analytics': True, 'version': '1.0'}
        gdpr_service.record_consent(session_id, consent_data)
        
        # Step 2: Get consent status
        consent_status = gdpr_service.get_consent_status(session_id)
        assert isinstance(consent_status, dict)
        
        # Step 3: Handle data access request
        access_result = gdpr_service.handle_data_access_request(session_id, email)
        assert access_result['status'] == 'success'
        
        # Step 4: Handle data deletion request
        deletion_result = gdpr_service.handle_data_deletion_request(session_id, email)
        assert deletion_result['status'] == 'success'
        
        # Step 5: Cleanup expired data
        cleanup_summary = gdpr_service.cleanup_expired_data()
        assert isinstance(cleanup_summary, dict)
