"""
GDPR Compliance Service for AI Istanbul

This module handles all GDPR-related operations including:
- Data access requests
- Data deletion requests
- Data portability
- Consent management
- Audit logging
- Personal data inventory
"""

import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from sqlalchemy import text, create_engine
from sqlalchemy.orm import Session
from database import SessionLocal, engine
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GDPRService:
    """
    Comprehensive GDPR compliance service
    """
    
    def __init__(self):
        self.retention_periods = {
            'chat_sessions': 30,  # days
            'user_feedback': 365,  # days  
            'analytics_data': 1095,  # 3 years
            'consent_records': 2555,  # 7 years (legal requirement)
            'audit_logs': 2555  # 7 years
        }
        
        self.personal_data_categories = {
            'session_data': 'Session identifiers and conversation history',
            'technical_data': 'IP addresses, browser information, device identifiers',
            'usage_data': 'Page views, feature usage, timestamps',
            'feedback_data': 'User feedback and ratings',
            'consent_data': 'Cookie preferences and consent records'
        }
    
    def create_audit_log(self, action: str, data_subject: str, details: Dict[str, Any]) -> None:
        """Create an audit log entry for GDPR compliance"""
        db = None
        try:
            db = SessionLocal()
            audit_entry = {
                'timestamp': datetime.now().isoformat(),
                'action': action,
                'data_subject': self._hash_identifier(data_subject),
                'details': json.dumps(details),
                'processor': 'ai_istanbul_system'
            }
            
            # Store in audit log table
            db.execute(text("""
                INSERT INTO gdpr_audit_log 
                (timestamp, action, data_subject_hash, details, processor)
                VALUES (:timestamp, :action, :data_subject, :details, :processor)
            """), audit_entry)
            db.commit()
            
            logger.info(f"GDPR audit log created: {action} for subject {data_subject[:8]}...")
            
        except Exception as e:
            logger.error(f"Failed to create audit log: {e}")
        finally:
            if db:
                db.close()
    
    def _hash_identifier(self, identifier: str) -> str:
        """Hash identifiers for privacy protection in logs"""
        return hashlib.sha256(identifier.encode()).hexdigest()[:16]
    
    def handle_data_access_request(self, session_id: str, email: Optional[str] = None) -> Dict[str, Any]:
        """Handle GDPR Article 15 - Right of access"""
        try:
            self.create_audit_log('data_access_request', session_id, {
                'email': email,
                'request_time': datetime.now().isoformat()
            })
            
            user_data = self._collect_user_data(session_id)
            
            # Prepare data export
            export_data = {
                'request_info': {
                    'session_id': session_id,
                    'request_date': datetime.now().isoformat(),
                    'data_controller': 'AI Istanbul Travel Assistant',
                    'contact_email': 'privacy@ai-istanbul.com'
                },
                'personal_data_categories': self.personal_data_categories,
                'retention_periods': self.retention_periods,
                'user_data': user_data,
                'legal_basis': 'Legitimate interest for travel recommendations',
                'data_sources': [
                    'User interactions with AI chatbot',
                    'Google Analytics (anonymized)',
                    'User feedback submissions',
                    'Browser session data'
                ]
            }
            
            if email:
                self._send_data_export_email(email, export_data)
            
            return {
                'status': 'success',
                'message': 'Data access request processed',
                'data': export_data
            }
            
        except Exception as e:
            logger.error(f"Data access request failed: {e}")
            return {
                'status': 'error',
                'message': f'Failed to process data access request: {str(e)}'
            }
    
    def handle_data_deletion_request(self, session_id: str, email: Optional[str] = None) -> Dict[str, Any]:
        """Handle GDPR Article 17 - Right to erasure"""
        try:
            self.create_audit_log('data_deletion_request', session_id, {
                'email': email,
                'request_time': datetime.now().isoformat()
            })
            
            deletion_summary = self._delete_user_data(session_id)
            
            if email:
                self._send_deletion_confirmation_email(email, deletion_summary)
            
            self.create_audit_log('data_deletion_completed', session_id, deletion_summary)
            
            return {
                'status': 'success',
                'message': 'Data deletion request processed',
                'deletion_summary': deletion_summary
            }
            
        except Exception as e:
            logger.error(f"Data deletion request failed: {e}")
            return {
                'status': 'error',
                'message': f'Failed to process data deletion request: {str(e)}'
            }
    
    def _collect_user_data(self, session_id: str) -> Dict[str, Any]:
        """Collect all user data associated with a session"""
        db = None
        try:
            db = SessionLocal()
            user_data = {}
            
            # Chat history
            chat_history = db.execute(text("""
                SELECT timestamp, user_message, ai_response, language
                FROM chat_history 
                WHERE session_id = :session_id
                ORDER BY timestamp DESC
            """), {'session_id': session_id}).fetchall()
            
            user_data['chat_history'] = [
                {
                    'timestamp': row[0],
                    'user_message': row[1],
                    'ai_response': row[2][:200] + '...' if len(row[2]) > 200 else row[2],
                    'language': row[3]
                }
                for row in chat_history
            ]
            
            # User feedback
            feedback_data = db.execute(text("""
                SELECT timestamp, feedback_type, rating, comment
                FROM user_feedback
                WHERE session_id = :session_id
                ORDER BY timestamp DESC
            """), {'session_id': session_id}).fetchall()
            
            user_data['feedback'] = [
                {
                    'timestamp': row[0],
                    'feedback_type': row[1],
                    'rating': row[2],
                    'comment': row[3]
                }
                for row in feedback_data
            ]
            
            # Session metadata - using existing schema
            session_data = db.execute(text("""
                SELECT created_at, last_activity, user_ip, user_agent
                FROM user_sessions
                WHERE session_id = :session_id
            """), {'session_id': session_id}).fetchone()
            
            if session_data:
                user_data['session_info'] = {
                    'created_at': session_data[0],
                    'last_activity': session_data[1],
                    'ip_address': session_data[2],
                    'user_agent': session_data[3],
                    'language_preference': 'en'  # Default since not in existing schema
                }
            
            # Consent records
            consent_data = db.execute(text("""
                SELECT consent_type, granted, timestamp, consent_version
                FROM user_consent
                WHERE session_id = :session_id
                ORDER BY timestamp DESC
            """), {'session_id': session_id}).fetchall()
            
            user_data['consent_records'] = [
                {
                    'consent_type': row[0],
                    'granted': row[1],
                    'timestamp': row[2],
                    'version': row[3]
                }
                for row in consent_data
            ]
            
            return user_data
            
        except Exception as e:
            logger.error(f"Failed to collect user data: {e}")
            return {}
        finally:
            if db:
                db.close()
    
    def _delete_user_data(self, session_id: str) -> Dict[str, Any]:
        """Delete all user data associated with a session"""
        db = None
        try:
            db = SessionLocal()
            deletion_summary = {
                'session_id': session_id,
                'deletion_timestamp': datetime.now().isoformat(),
                'deleted_records': {}
            }
            
            # Delete chat history
            chat_result = db.execute(text("""
                DELETE FROM chat_history WHERE session_id = :session_id
            """), {'session_id': session_id})
            chat_deleted = getattr(chat_result, 'rowcount', 0)
            deletion_summary['deleted_records']['chat_history'] = chat_deleted
            
            # Delete user feedback
            feedback_result = db.execute(text("""
                DELETE FROM user_feedback WHERE session_id = :session_id
            """), {'session_id': session_id})
            feedback_deleted = getattr(feedback_result, 'rowcount', 0)
            deletion_summary['deleted_records']['feedback'] = feedback_deleted
            
            # Delete session data (keep minimal audit trail) - using existing schema
            session_result = db.execute(text("""
                UPDATE user_sessions 
                SET user_ip = 'DELETED', user_agent = 'DELETED'
                WHERE session_id = :session_id
            """), {'session_id': session_id})
            session_deleted = getattr(session_result, 'rowcount', 0)
            deletion_summary['deleted_records']['session_data'] = session_deleted
            
            # Mark consent records as deleted (keep for legal compliance)
            consent_result = db.execute(text("""
                UPDATE user_consent 
                SET deleted_at = :now
                WHERE session_id = :session_id
            """), {'session_id': session_id, 'now': datetime.now()})
            consent_updated = getattr(consent_result, 'rowcount', 0)
            deletion_summary['deleted_records']['consent_records'] = consent_updated
            
            db.commit()
            
            return deletion_summary
            
        except Exception as e:
            logger.error(f"Failed to delete user data: {e}")
            raise e
        finally:
            if db:
                db.close()
    
    def record_consent(self, session_id: str, consent_data: Dict[str, Any]) -> None:
        """Record user consent for GDPR compliance"""
        db = None
        try:
            db = SessionLocal()
            
            for consent_type, granted in consent_data.items():
                if consent_type != 'timestamp' and consent_type != 'version':
                    db.execute(text("""
                        INSERT INTO user_consent 
                        (session_id, consent_type, granted, timestamp, consent_version)
                        VALUES (:session_id, :consent_type, :granted, :timestamp, :version)
                    """), {
                        'session_id': session_id,
                        'consent_type': consent_type,
                        'granted': granted,
                        'timestamp': datetime.now(),
                        'version': consent_data.get('version', '1.0')
                    })
            
            db.commit()
            
            self.create_audit_log('consent_recorded', session_id, consent_data)
            
        except Exception as e:
            logger.error(f"Failed to record consent: {e}")
        finally:
            if db:
                db.close()
    
    def cleanup_expired_data(self) -> Dict[str, int]:
        """Clean up expired data according to retention policies"""
        db = None
        try:
            db = SessionLocal()
            cleanup_summary = {}
            
            for data_type, retention_days in self.retention_periods.items():
                if data_type == 'consent_records' or data_type == 'audit_logs':
                    continue  # Don't auto-delete these for legal compliance
                
                cutoff_date = datetime.now() - timedelta(days=retention_days)
                deleted = 0
                
                if data_type == 'chat_sessions':
                    result = db.execute(text("""
                        DELETE FROM chat_history 
                        WHERE timestamp < :cutoff_date
                    """), {'cutoff_date': cutoff_date})
                    deleted = getattr(result, 'rowcount', 0)
                    
                elif data_type == 'user_feedback':
                    result = db.execute(text("""
                        DELETE FROM user_feedback 
                        WHERE timestamp < :cutoff_date
                    """), {'cutoff_date': cutoff_date})
                    deleted = getattr(result, 'rowcount', 0)
                
                cleanup_summary[data_type] = deleted
            
            db.commit()
            
            self.create_audit_log('automated_data_cleanup', 'system', cleanup_summary)
            
            return cleanup_summary
            
        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")
            return {}
        finally:
            if db:
                db.close()
    
    def _send_data_export_email(self, email: str, export_data: Dict[str, Any]) -> None:
        """Send data export notification (simplified for GDPR compliance)"""
        try:
            # For production, implement actual email sending
            # For now, log the request for manual processing
            logger.info(f"Data export requested for email: {email}")
            logger.info(f"Export data summary: {len(export_data.get('user_data', {}))} data categories")
            
            # In production, you would:
            # 1. Send email with secure download link
            # 2. Store encrypted export file temporarily
            # 3. Send notification to data protection officer
            
        except Exception as e:
            logger.error(f"Failed to process data export email: {e}")
    
    def _send_deletion_confirmation_email(self, email: str, deletion_summary: Dict[str, Any]) -> None:
        """Send deletion confirmation notification"""
        try:
            logger.info(f"Data deletion completed for email: {email}")
            logger.info(f"Deletion summary: {deletion_summary}")
            
            # In production, send confirmation email
            
        except Exception as e:
            logger.error(f"Failed to process deletion confirmation email: {e}")
    
    def get_consent_status(self, session_id: str) -> Dict[str, Any]:
        """Get current consent status for a session"""
        db = None
        try:
            db = SessionLocal()
            
            consent_data = db.execute(text("""
                SELECT consent_type, granted, timestamp 
                FROM user_consent 
                WHERE session_id = :session_id AND deleted_at IS NULL
                ORDER BY timestamp DESC
            """), {'session_id': session_id}).fetchall()
            
            consent_status = {}
            for row in consent_data:
                if row[0] not in consent_status:  # Get latest consent for each type
                    consent_status[row[0]] = {
                        'granted': row[1],
                        'timestamp': row[2]
                    }
            
            return consent_status
            
        except Exception as e:
            logger.error(f"Failed to get consent status: {e}")
            return {}
        finally:
            if db:
                db.close()

# Global instance
gdpr_service = GDPRService()

def init_gdpr_tables():
    """Initialize GDPR-related database tables"""
    db = None
    try:
        db = SessionLocal()
        
        # Create GDPR audit log table
        db.execute(text("""
            CREATE TABLE IF NOT EXISTS gdpr_audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                action TEXT NOT NULL,
                data_subject_hash TEXT NOT NULL,
                details TEXT,
                processor TEXT NOT NULL
            )
        """))
        
        # Create user consent table
        db.execute(text("""
            CREATE TABLE IF NOT EXISTS user_consent (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                consent_type TEXT NOT NULL,
                granted BOOLEAN NOT NULL,
                timestamp DATETIME NOT NULL,
                consent_version TEXT DEFAULT '1.0',
                deleted_at DATETIME NULL
            )
        """))
        
        # Create user sessions table
        db.execute(text("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                created_at DATETIME NOT NULL,
                last_activity DATETIME NOT NULL,
                ip_address TEXT,
                user_agent TEXT,
                language_preference TEXT DEFAULT 'en',
                deleted_at DATETIME NULL
            )
        """))
        
        # Create user feedback table
        db.execute(text("""
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                feedback_type TEXT NOT NULL,
                rating INTEGER,
                comment TEXT,
                deleted_at DATETIME NULL
            )
        """))
        
        # Create chat history table
        db.execute(text("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                user_message TEXT NOT NULL,
                ai_response TEXT NOT NULL,
                language TEXT DEFAULT 'en',
                deleted_at DATETIME NULL
            )
        """))
        
        db.commit()
        logger.info("GDPR database tables initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize GDPR tables: {e}")
    finally:
        if db:
            db.close()

# Initialize tables on import
init_gdpr_tables()
