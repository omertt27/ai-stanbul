"""
Intent Feedback Database Model
Stores user feedback on intent classification for active learning
"""

from sqlalchemy import Column, Integer, String, Text, Float, Boolean, DateTime, Index
from datetime import datetime
from typing import Optional, Dict, Any
import json

# Import Base from parent database module
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from database import Base


class IntentFeedback(Base):
    """
    Store user feedback on intent classification
    Used for active learning and model improvement
    """
    __tablename__ = "intent_feedback"
    __table_args__ = {'extend_existing': True}  # Allow table redefinition
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Session tracking
    session_id = Column(String(100), nullable=False, index=True)
    user_id = Column(String(100), nullable=True, index=True)
    
    # Query information
    original_query = Column(Text, nullable=False)
    language = Column(String(10), index=True)  # 'tr', 'en', 'mixed'
    
    # Classification results
    predicted_intent = Column(String(50), nullable=False, index=True)
    predicted_confidence = Column(Float, nullable=False)
    classification_method = Column(String(20))  # 'neural', 'fallback'
    latency_ms = Column(Float)
    
    # User feedback
    is_correct = Column(Boolean, nullable=True, index=True)  # User confirmed correct/wrong
    actual_intent = Column(String(50), nullable=True, index=True)  # If wrong, what should it be
    feedback_type = Column(String(20), index=True)  # 'explicit', 'implicit', 'click'
    
    # Metadata
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    user_action = Column(String(100))  # What did user do after classification
    
    # Analysis flags
    used_for_training = Column(Boolean, default=False, index=True)
    review_status = Column(String(20), default='pending', index=True)  # 'pending', 'reviewed', 'approved'
    
    # Additional context (JSON)
    context_data = Column(Text)  # Stores JSON with additional context
    
    # Indexes for common queries
    __table_args__ = (
        Index('idx_feedback_status', 'review_status', 'used_for_training'),
        Index('idx_feedback_quality', 'feedback_type', 'is_correct', 'timestamp'),
        Index('idx_training_data', 'used_for_training', 'review_status', 'predicted_intent'),
    )
    
    def __repr__(self):
        return f"<IntentFeedback(id={self.id}, query='{self.original_query[:30]}...', intent={self.predicted_intent}, correct={self.is_correct})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'session_id': self.session_id,
            'user_id': self.user_id,
            'original_query': self.original_query,
            'language': self.language,
            'predicted_intent': self.predicted_intent,
            'predicted_confidence': self.predicted_confidence,
            'classification_method': self.classification_method,
            'latency_ms': self.latency_ms,
            'is_correct': self.is_correct,
            'actual_intent': self.actual_intent,
            'feedback_type': self.feedback_type,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'user_action': self.user_action,
            'used_for_training': self.used_for_training,
            'review_status': self.review_status,
            'context_data': json.loads(self.context_data) if self.context_data else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IntentFeedback':
        """Create from dictionary"""
        feedback = cls()
        
        # Set attributes
        for key, value in data.items():
            if key == 'context_data' and value:
                setattr(feedback, key, json.dumps(value))
            elif key == 'timestamp' and isinstance(value, str):
                setattr(feedback, key, datetime.fromisoformat(value))
            elif hasattr(feedback, key):
                setattr(feedback, key, value)
        
        return feedback
    
    def mark_for_training(self):
        """Mark this feedback as ready for training"""
        self.used_for_training = True
        self.review_status = 'approved'
    
    def is_high_quality(self) -> bool:
        """Check if this feedback is high quality for training"""
        # Explicit feedback with clear correction
        if self.feedback_type == 'explicit' and self.is_correct is not None:
            return True
        
        # Implicit feedback with high confidence
        if self.feedback_type == 'implicit' and self.predicted_confidence > 0.8:
            return True
        
        return False


class FeedbackStatistics:
    """Helper class for feedback statistics"""
    
    @staticmethod
    def calculate_accuracy(session, start_date: Optional[datetime] = None, 
                          end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Calculate accuracy metrics from feedback"""
        query = session.query(IntentFeedback).filter(
            IntentFeedback.is_correct.isnot(None)
        )
        
        if start_date:
            query = query.filter(IntentFeedback.timestamp >= start_date)
        if end_date:
            query = query.filter(IntentFeedback.timestamp <= end_date)
        
        feedback_items = query.all()
        
        if not feedback_items:
            return {
                'total': 0,
                'correct': 0,
                'incorrect': 0,
                'accuracy': 0.0
            }
        
        total = len(feedback_items)
        correct = sum(1 for item in feedback_items if item.is_correct)
        
        return {
            'total': total,
            'correct': correct,
            'incorrect': total - correct,
            'accuracy': correct / total if total > 0 else 0.0
        }
    
    @staticmethod
    def get_intent_distribution(session, feedback_type: Optional[str] = None) -> Dict[str, int]:
        """Get distribution of intents from feedback"""
        query = session.query(
            IntentFeedback.predicted_intent,
            IntentFeedback
        )
        
        if feedback_type:
            query = query.filter(IntentFeedback.feedback_type == feedback_type)
        
        distribution = {}
        for intent, _ in query.all():
            distribution[intent] = distribution.get(intent, 0) + 1
        
        return distribution
    
    @staticmethod
    def get_training_ready_count(session) -> int:
        """Count feedback items ready for training"""
        return session.query(IntentFeedback).filter(
            IntentFeedback.review_status == 'approved',
            IntentFeedback.used_for_training == False
        ).count()


# Create tables function
def create_tables(engine):
    """Create all tables in the database"""
    Base.metadata.create_all(engine)
    print("✅ Intent feedback tables created successfully")


if __name__ == "__main__":
    # Test the model
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    # Create in-memory database for testing
    engine = create_engine('sqlite:///:memory:', echo=True)
    create_tables(engine)
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Test creating feedback
    feedback = IntentFeedback(
        session_id='test_session_123',
        user_id='user_456',
        original_query='Sultanahmet\'te restoran öner',
        language='tr',
        predicted_intent='restaurant',
        predicted_confidence=0.85,
        classification_method='neural',
        latency_ms=12.5,
        feedback_type='explicit',
        is_correct=True
    )
    
    session.add(feedback)
    session.commit()
    
    print(f"\n✅ Test feedback created: {feedback}")
    print(f"📊 Is high quality: {feedback.is_high_quality()}")
    print(f"📋 Dict representation: {feedback.to_dict()}")
    
    session.close()
