"""
Comprehensive tests for GDPR Service
Tests data protection, privacy compliance, and user rights
"""
import pytest
import asyncio
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List
from datetime import datetime, timedelta

class TestGDPRService:
    """Test GDPR Service functionality."""
    
    @pytest.fixture
    def mock_database(self):
        """Mock database for testing."""
        mock_db = MagicMock()
        mock_db.query.return_value = mock_db
        mock_db.filter.return_value = mock_db
        mock_db.all.return_value = []
        mock_db.first.return_value = None
        mock_db.delete.return_value = None
        mock_db.commit.return_value = None
        return mock_db
    
    @pytest.fixture
    def gdpr_service(self, mock_database):
        """Create GDPR service instance."""
        with patch('backend.gdpr_service.get_db_session', return_value=mock_database):
            from backend.gdpr_service import GDPRService
            return GDPRService()
    
    def test_gdpr_service_initialization(self, gdpr_service):
        """Test GDPR service initializes correctly."""
        assert gdpr_service is not None
        assert hasattr(gdpr_service, 'handle_data_access_request')
        assert hasattr(gdpr_service, 'handle_data_deletion_request')
        assert hasattr(gdpr_service, 'get_user_consent_status')
    
    def test_data_access_request_valid_user(self, gdpr_service, mock_database):
        """Test handling valid data access request (Article 15)."""
        session_id = "test-session-123"
        email = "user@example.com"
        
        # Mock user data
        mock_user_data = [
            MagicMock(session_id=session_id, query="Best restaurants", timestamp=datetime.now()),
            MagicMock(session_id=session_id, query="Museums in Istanbul", timestamp=datetime.now())
        ]
        mock_database.all.return_value = mock_user_data
        
        result = gdpr_service.handle_data_access_request(session_id, email)
        
        assert result["status"] == "success"
        assert "data" in result
        assert len(result["data"]["queries"]) == 2
        assert "personal_data" in result["data"]
        assert result["data"]["request_fulfilled"] is True
    
    def test_data_access_request_no_data(self, gdpr_service, mock_database):
        """Test data access request when no data exists."""
        session_id = "non-existent-session"
        email = "newuser@example.com"
        
        mock_database.all.return_value = []
        
        result = gdpr_service.handle_data_access_request(session_id, email)
        
        assert result["status"] == "success"
        assert result["data"]["queries"] == []
        assert result["data"]["message"] == "No personal data found for this session"
    
    def test_data_deletion_request_success(self, gdpr_service, mock_database):
        """Test successful data deletion request (Article 17)."""
        session_id = "test-session-456"
        email = "user@example.com"
        
        # Mock existing data
        mock_database.all.return_value = [MagicMock(), MagicMock()]  # 2 records
        
        result = gdpr_service.handle_data_deletion_request(session_id, email)
        
        assert result["status"] == "success"
        assert "deletion_completed" in result
        assert result["deletion_completed"] is True
        assert "deleted_records" in result
        mock_database.delete.assert_called()
        mock_database.commit.assert_called()
    
    def test_data_deletion_request_no_data(self, gdpr_service, mock_database):
        """Test deletion request when no data exists."""
        session_id = "empty-session"
        email = "user@example.com"
        
        mock_database.all.return_value = []
        
        result = gdpr_service.handle_data_deletion_request(session_id, email)
        
        assert result["status"] == "success"
        assert result["message"] == "No data found to delete for this session"
        assert result["deleted_records"] == 0
    
    def test_consent_management_grant_consent(self, gdpr_service, mock_database):
        """Test granting user consent."""
        session_id = "consent-test-session"
        consent_types = ["data_processing", "analytics", "marketing"]
        
        result = gdpr_service.grant_consent(session_id, consent_types)
        
        assert result["status"] == "success"
        assert result["consent_granted"] is True
        assert set(result["consent_types"]) == set(consent_types)
        assert "timestamp" in result
    
    def test_consent_management_withdraw_consent(self, gdpr_service, mock_database):
        """Test withdrawing user consent."""
        session_id = "consent-withdrawal-session"
        consent_types = ["marketing", "analytics"]
        
        result = gdpr_service.withdraw_consent(session_id, consent_types)
        
        assert result["status"] == "success"
        assert result["consent_withdrawn"] is True
        assert set(result["withdrawn_types"]) == set(consent_types)
    
    def test_get_consent_status(self, gdpr_service, mock_database):
        """Test retrieving user consent status."""
        session_id = "status-check-session"
        
        # Mock consent data
        mock_consent = MagicMock()
        mock_consent.data_processing = True
        mock_consent.analytics = False
        mock_consent.marketing = True
        mock_database.first.return_value = mock_consent
        
        result = gdpr_service.get_user_consent_status(session_id)
        
        assert result["status"] == "success"
        assert result["consent"]["data_processing"] is True
        assert result["consent"]["analytics"] is False
        assert result["consent"]["marketing"] is True
    
    def test_data_portability_request(self, gdpr_service, mock_database):
        """Test data portability request (Article 20)."""
        session_id = "portability-test-session"
        email = "user@example.com"
        export_format = "json"
        
        # Mock user data
        mock_data = [
            MagicMock(query="Restaurant query", response="Restaurant response", timestamp=datetime.now()),
            MagicMock(query="Museum query", response="Museum response", timestamp=datetime.now())
        ]
        mock_database.all.return_value = mock_data
        
        result = gdpr_service.handle_data_portability_request(session_id, email, export_format)
        
        assert result["status"] == "success"
        assert "export_data" in result
        assert result["format"] == "json"
        assert len(result["export_data"]["interactions"]) == 2
    
    def test_data_minimization_compliance(self, gdpr_service):
        """Test data minimization principles."""
        # Test data collection minimization
        essential_data = {
            "query": "Best restaurants",
            "session_id": "test-session",
            "timestamp": datetime.now()
        }
        
        non_essential_data = {
            "user_agent": "Mozilla/5.0...",
            "ip_address": "192.168.1.1",
            "device_fingerprint": "abc123"
        }
        
        minimized_data = gdpr_service.minimize_data_collection(
            essential_data, 
            non_essential_data
        )
        
        # Should only include essential data
        assert "query" in minimized_data
        assert "session_id" in minimized_data
        assert "timestamp" in minimized_data
        assert "device_fingerprint" not in minimized_data
    
    def test_data_retention_compliance(self, gdpr_service, mock_database):
        """Test data retention policy compliance."""
        # Mock old data that should be deleted
        old_timestamp = datetime.now() - timedelta(days=400)  # Over retention period
        recent_timestamp = datetime.now() - timedelta(days=30)  # Within retention period
        
        mock_old_data = [
            MagicMock(timestamp=old_timestamp, session_id="old-session-1"),
            MagicMock(timestamp=old_timestamp, session_id="old-session-2")
        ]
        mock_recent_data = [
            MagicMock(timestamp=recent_timestamp, session_id="recent-session")
        ]
        
        mock_database.filter.return_value.all.side_effect = [mock_old_data, mock_recent_data]
        
        result = gdpr_service.cleanup_expired_data()
        
        assert result["status"] == "success"
        assert result["deleted_records"] == 2
        assert "retention_policy_applied" in result
    
    def test_privacy_impact_assessment(self, gdpr_service):
        """Test privacy impact assessment for new features."""
        feature_description = {
            "name": "Enhanced Analytics",
            "data_types": ["user_queries", "session_data", "preferences"],
            "processing_purpose": "Improve AI responses",
            "data_sharing": False,
            "retention_period": "12 months"
        }
        
        assessment = gdpr_service.conduct_privacy_impact_assessment(feature_description)
        
        assert assessment["status"] == "completed"
        assert "risk_level" in assessment
        assert assessment["risk_level"] in ["low", "medium", "high"]
        assert "recommendations" in assessment
        assert "compliance_status" in assessment
    
    def test_data_breach_notification(self, gdpr_service):
        """Test data breach notification procedures."""
        breach_details = {
            "incident_type": "unauthorized_access",
            "affected_users": 150,
            "data_types": ["session_ids", "query_history"],
            "discovery_date": datetime.now(),
            "containment_measures": ["Access revoked", "Passwords reset"]
        }
        
        notification = gdpr_service.handle_data_breach_notification(breach_details)
        
        assert notification["status"] == "processed"
        assert notification["notification_required"] is True  # >150 users affected
        assert "authority_notification_deadline" in notification
        assert "user_notification_required" in notification
    
    def test_data_subject_rights_validation(self, gdpr_service):
        """Test validation of data subject rights requests."""
        # Test valid request
        valid_request = {
            "request_type": "access",
            "session_id": "valid-session-123",
            "email": "user@example.com",
            "verification_data": {"ip_address": "192.168.1.1"}
        }
        
        validation_result = gdpr_service.validate_data_subject_request(valid_request)
        assert validation_result["valid"] is True
        assert validation_result["request_type"] == "access"
        
        # Test invalid request
        invalid_request = {
            "request_type": "invalid_type",
            "session_id": "",
            "email": "invalid_email"
        }
        
        validation_result = gdpr_service.validate_data_subject_request(invalid_request)
        assert validation_result["valid"] is False
        assert "errors" in validation_result
    
    def test_anonymization_pseudonymization(self, gdpr_service):
        """Test data anonymization and pseudonymization."""
        personal_data = {
            "session_id": "user-session-123",
            "email": "user@example.com",
            "ip_address": "192.168.1.100",
            "queries": ["Best restaurants near me", "Hotels in Sultanahmet"]
        }
        
        # Test pseudonymization (reversible)
        pseudonymized = gdpr_service.pseudonymize_data(personal_data)
        assert pseudonymized["session_id"] != personal_data["session_id"]
        assert pseudonymized["email"] != personal_data["email"]
        assert "queries" in pseudonymized  # Non-personal data preserved
        
        # Test anonymization (irreversible)
        anonymized = gdpr_service.anonymize_data(personal_data)
        assert "session_id" not in anonymized
        assert "email" not in anonymized
        assert "ip_address" not in anonymized
        assert "queries" in anonymized  # Non-personal data preserved
    
    def test_cross_border_data_transfer_compliance(self, gdpr_service):
        """Test compliance with cross-border data transfer rules."""
        transfer_request = {
            "destination_country": "US",
            "data_types": ["user_preferences", "session_data"],
            "purpose": "AI model training",
            "safeguards": ["Standard Contractual Clauses"]
        }
        
        compliance_check = gdpr_service.validate_cross_border_transfer(transfer_request)
        
        assert compliance_check["status"] == "evaluated"
        assert "transfer_allowed" in compliance_check
        assert "required_safeguards" in compliance_check
        assert "adequacy_decision" in compliance_check
    
    def test_audit_trail_logging(self, gdpr_service):
        """Test audit trail logging for GDPR operations."""
        # Test logging of data access
        gdpr_service.log_gdpr_operation(
            operation_type="data_access",
            session_id="audit-test-session",
            details={"requester_email": "user@example.com"}
        )
        
        # Test logging of data deletion
        gdpr_service.log_gdpr_operation(
            operation_type="data_deletion",
            session_id="audit-test-session",
            details={"deleted_records": 5}
        )
        
        # Retrieve audit trail
        audit_trail = gdpr_service.get_audit_trail("audit-test-session")
        
        assert len(audit_trail) == 2
        assert audit_trail[0]["operation_type"] == "data_access"
        assert audit_trail[1]["operation_type"] == "data_deletion"
        assert all("timestamp" in entry for entry in audit_trail)
    
    def test_error_handling_invalid_session(self, gdpr_service, mock_database):
        """Test error handling for invalid session IDs."""
        invalid_session_id = None
        email = "user@example.com"
        
        result = gdpr_service.handle_data_access_request(invalid_session_id, email)
        
        assert result["status"] == "error"
        assert "Invalid session ID" in result["message"]
    
    def test_error_handling_database_failure(self, gdpr_service, mock_database):
        """Test error handling when database operations fail."""
        session_id = "db-error-session"
        email = "user@example.com"
        
        # Mock database error
        mock_database.all.side_effect = Exception("Database connection failed")
        
        result = gdpr_service.handle_data_access_request(session_id, email)
        
        assert result["status"] == "error"
        assert "database error" in result["message"].lower()
    
    @pytest.mark.asyncio
    async def test_async_gdpr_operations(self, gdpr_service, mock_database):
        """Test asynchronous GDPR operations for performance."""
        session_ids = [f"async-session-{i}" for i in range(5)]
        email = "user@example.com"
        
        # Test concurrent data access requests
        tasks = [
            gdpr_service.handle_data_access_request_async(session_id, email)
            for session_id in session_ids
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        assert len(results) == 5
        for result in results:
            assert not isinstance(result, Exception)
            assert result["status"] in ["success", "error"]
