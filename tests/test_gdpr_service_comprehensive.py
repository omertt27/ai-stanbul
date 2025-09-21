"""
Comprehensive tests for GDPR Service - ACTIVELY USED MODULE
Tests GDPR compliance operations, data management, and audit logging
"""
import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from backend.gdpr_service import GDPRService


class TestGDPRService:
    """Test the GDPR compliance service functionality."""
    
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
        assert 'chat_sessions' in gdpr_service.retention_periods
        assert 'session_data' in gdpr_service.personal_data_categories
    
    def test_retention_periods_configuration(self, gdpr_service):
        """Test retention periods are properly configured."""
        assert gdpr_service.retention_periods['chat_sessions'] == 30
        assert gdpr_service.retention_periods['user_feedback'] == 365
        assert gdpr_service.retention_periods['analytics_data'] == 1095
        assert gdpr_service.retention_periods['consent_records'] == 2555
        assert gdpr_service.retention_periods['audit_logs'] == 2555
    
    def test_personal_data_categories(self, gdpr_service):
        """Test personal data categories are defined."""
        categories = gdpr_service.personal_data_categories
        assert 'session_data' in categories
        assert 'technical_data' in categories
        assert 'usage_data' in categories
        assert 'feedback_data' in categories
        assert 'consent_data' in categories
    
    def test_hash_identifier(self, gdpr_service):
        """Test identifier hashing for privacy protection."""
        identifier = "test_session_123"
        hashed = gdpr_service._hash_identifier(identifier)
        
        assert len(hashed) == 16  # Should be truncated to 16 chars
        assert hashed != identifier  # Should be different from original
        
        # Same input should produce same hash
        hashed2 = gdpr_service._hash_identifier(identifier)
        assert hashed == hashed2
        
        # Different input should produce different hash
        hashed3 = gdpr_service._hash_identifier("different_session")
        assert hashed != hashed3
    
    def test_create_audit_log_success(self, gdpr_service, mock_db_session):
        """Test successful audit log creation."""
        action = "data_access_request"
        data_subject = "session_123"
        details = {"requested_data": "chat_history", "status": "completed"}
        
        gdpr_service.create_audit_log(action, data_subject, details)
        
        # Verify database operations
        mock_db_session.execute.assert_called_once()
        mock_db_session.commit.assert_called_once()
        mock_db_session.close.assert_called_once()
        
        # Check the SQL call
        call_args = mock_db_session.execute.call_args
        assert "INSERT INTO gdpr_audit_log" in str(call_args[0][0])
        
        # Check parameters
        params = call_args[1]
        assert params['action'] == action
        assert params['processor'] == 'ai_istanbul_system'
        assert 'timestamp' in params
    
    def test_create_audit_log_database_error(self, gdpr_service, mock_db_session):
        """Test audit log creation with database error."""
        mock_db_session.execute.side_effect = Exception("Database error")
        
        action = "data_deletion"
        data_subject = "session_456"
        details = {"deleted_items": 5}
        
        # Should not raise exception, just log error
        gdpr_service.create_audit_log(action, data_subject, details)
        
        # Should still try to close the session
        mock_db_session.close.assert_called_once()
    
    def test_process_access_request(self, gdpr_service, mock_db_session):
        """Test data access request processing."""
        session_id = "test_session_789"
        
        # Mock database responses
        mock_db_session.execute.return_value.fetchall.side_effect = [
            [("query1", "response1", "2024-01-01")],  # interactions
            [("Great service!", 5, "2024-01-01")],    # feedback
            [(10, "2024-01-01")]                      # analytics
        ]
        
        result = gdpr_service.process_access_request(session_id)
        
        assert result["status"] == "success"
        assert "data" in result
        assert "interactions" in result["data"]
        assert "feedback" in result["data"]
        assert "analytics" in result["data"]
        
        # Should create audit log
        assert mock_db_session.execute.call_count >= 4  # 3 queries + 1 audit log
    
    def test_process_deletion_request(self, gdpr_service, mock_db_session):
        """Test data deletion request processing."""
        session_id = "test_session_delete"
        
        # Mock deletion results
        mock_db_session.execute.return_value.rowcount = 5
        
        result = gdpr_service.process_deletion_request(session_id)
        
        assert result["status"] == "success"
        assert result["deleted_items"] > 0
        
        # Should execute multiple DELETE statements
        assert mock_db_session.execute.call_count >= 3
        mock_db_session.commit.assert_called()
    
    def test_export_personal_data(self, gdpr_service, mock_db_session):
        """Test personal data export for portability."""
        session_id = "export_test_session"
        
        # Mock data for export
        mock_db_session.execute.return_value.fetchall.side_effect = [
            [("query1", "response1", "2024-01-01")],
            [("feedback1", 4, "2024-01-01")],
            [(15, "session_metadata")]
        ]
        
        exported_data = gdpr_service.export_personal_data(session_id)
        
        assert "interactions" in exported_data
        assert "feedback" in exported_data
        assert "metadata" in exported_data
        assert "export_timestamp" in exported_data
        
        # Should be JSON serializable
        json_str = json.dumps(exported_data)
        assert len(json_str) > 0
    
    def test_check_consent_status(self, gdpr_service, mock_db_session):
        """Test consent status checking."""
        session_id = "consent_test_session"
        
        # Mock consent record
        mock_db_session.execute.return_value.fetchone.return_value = (
            True, True, False, "2024-01-01"
        )
        
        consent = gdpr_service.check_consent_status(session_id)
        
        assert consent["analytics_consent"] is True
        assert consent["marketing_consent"] is True
        assert consent["third_party_consent"] is False
        assert "consent_date" in consent
    
    def test_update_consent(self, gdpr_service, mock_db_session):
        """Test consent update functionality."""
        session_id = "consent_update_session"
        consent_data = {
            "analytics_consent": True,
            "marketing_consent": False,
            "third_party_consent": True
        }
        
        result = gdpr_service.update_consent(session_id, consent_data)
        
        assert result["status"] == "success"
        mock_db_session.execute.assert_called()
        mock_db_session.commit.assert_called()
    
    def test_cleanup_expired_data(self, gdpr_service, mock_db_session):
        """Test automatic cleanup of expired data."""
        # Mock cleanup results
        mock_db_session.execute.return_value.rowcount = 25
        
        result = gdpr_service.cleanup_expired_data()
        
        assert result["status"] == "success"
        assert result["total_deleted"] == 25 * 4  # 4 cleanup operations
        
        # Should execute cleanup for each data type
        assert mock_db_session.execute.call_count >= 4
        mock_db_session.commit.assert_called()
    
    def test_get_data_inventory(self, gdpr_service, mock_db_session):
        """Test personal data inventory generation."""
        # Mock inventory data
        mock_db_session.execute.return_value.fetchall.return_value = [
            ("interactions", 150),
            ("feedback", 25),
            ("consent_records", 1),
            ("audit_logs", 75)
        ]
        
        inventory = gdpr_service.get_data_inventory()
        
        assert "data_categories" in inventory
        assert "total_records" in inventory
        assert inventory["total_records"] == 251  # Sum of all records
        
        # Should include retention information
        assert "retention_policies" in inventory
    
    def test_generate_compliance_report(self, gdpr_service, mock_db_session):
        """Test GDPR compliance report generation."""
        # Mock report data
        mock_db_session.execute.return_value.fetchall.side_effect = [
            [("data_access_request", 15)],     # request types
            [("2024-01-01", 5)],               # daily activity
            [("completed", 18), ("pending", 2)] # request status
        ]
        
        report = gdpr_service.generate_compliance_report(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31)
        )
        
        assert "period" in report
        assert "request_summary" in report
        assert "daily_activity" in report
        assert "compliance_metrics" in report
    
    def test_validate_retention_compliance(self, gdpr_service, mock_db_session):
        """Test retention policy compliance validation."""
        # Mock overdue data
        mock_db_session.execute.return_value.fetchall.return_value = [
            ("interactions", 5, "2023-01-01"),
            ("feedback", 2, "2022-06-01")
        ]
        
        compliance = gdpr_service.validate_retention_compliance()
        
        assert "compliant" in compliance
        assert "overdue_items" in compliance
        assert len(compliance["overdue_items"]) == 2
        assert compliance["overdue_items"][0]["category"] == "interactions"
    
    def test_anonymize_session_data(self, gdpr_service, mock_db_session):
        """Test session data anonymization."""
        session_id = "anonymize_test_session"
        
        result = gdpr_service.anonymize_session_data(session_id)
        
        assert result["status"] == "success"
        assert "anonymized_items" in result
        
        # Should update records instead of deleting
        mock_db_session.execute.assert_called()
        mock_db_session.commit.assert_called()
    
    def test_handle_breach_notification(self, gdpr_service, mock_db_session):
        """Test data breach notification handling."""
        breach_details = {
            "incident_type": "unauthorized_access",
            "affected_sessions": ["session1", "session2"],
            "severity": "high",
            "description": "Potential data exposure"
        }
        
        result = gdpr_service.handle_breach_notification(breach_details)
        
        assert result["status"] == "logged"
        assert "notification_id" in result
        assert "authorities_notified" in result
        
        # Should create audit log for breach
        mock_db_session.execute.assert_called()
        mock_db_session.commit.assert_called()
    
    def test_get_processing_activities(self, gdpr_service):
        """Test processing activities documentation."""
        activities = gdpr_service.get_processing_activities()
        
        assert "chat_processing" in activities
        assert "analytics_processing" in activities
        assert "feedback_processing" in activities
        
        # Each activity should have required GDPR fields
        for activity in activities.values():
            assert "purpose" in activity
            assert "legal_basis" in activity
            assert "data_categories" in activity
            assert "retention_period" in activity
    
    def test_integration_full_gdpr_workflow(self, gdpr_service, mock_db_session):
        """Test complete GDPR workflow from request to completion."""
        session_id = "gdpr_integration_test"
        
        # 1. Check current consent
        mock_db_session.execute.return_value.fetchone.return_value = (
            True, False, False, "2024-01-01"
        )
        consent = gdpr_service.check_consent_status(session_id)
        assert consent["analytics_consent"] is True
        
        # 2. Process access request
        mock_db_session.execute.return_value.fetchall.side_effect = [
            [("query", "response", "2024-01-01")],
            [("feedback", 5, "2024-01-01")],
            [(1, "metadata")]
        ]
        access_result = gdpr_service.process_access_request(session_id)
        assert access_result["status"] == "success"
        
        # 3. Update consent
        new_consent = {"analytics_consent": False, "marketing_consent": False}
        consent_result = gdpr_service.update_consent(session_id, new_consent)
        assert consent_result["status"] == "success"
        
        # 4. Anonymize data
        anonymize_result = gdpr_service.anonymize_session_data(session_id)
        assert anonymize_result["status"] == "success"
        
        # Verify all operations were logged
        assert mock_db_session.execute.call_count >= 8  # Multiple operations
        assert mock_db_session.commit.call_count >= 4   # Each operation commits
