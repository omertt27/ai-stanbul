"""
User Feedback Collection and Analysis System
Collects ratings, comments, and analytics from users
"""
import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import redis
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of feedback"""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    STAR_RATING = "star_rating"
    COMMENT = "comment"
    BUG_REPORT = "bug_report"


@dataclass
class UserFeedback:
    """User feedback entry"""
    id: str
    session_id: str
    query: str
    response: str
    feedback_type: str
    rating: Optional[int]  # 1-5 for star ratings
    comment: Optional[str]
    language: str
    timestamp: float
    user_agent: Optional[str]
    metadata: Dict[str, Any]
    
    def to_dict(self):
        return asdict(self)


class FeedbackCollector:
    """Collect and analyze user feedback"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.feedback_key = "feedback:entries"
        self.analytics_key = "feedback:analytics"
    
    def submit_feedback(
        self,
        session_id: str,
        query: str,
        response: str,
        feedback_type: FeedbackType,
        rating: Optional[int] = None,
        comment: Optional[str] = None,
        language: str = "en",
        user_agent: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Submit user feedback
        
        Args:
            session_id: User session identifier
            query: User's query
            response: LLM's response
            feedback_type: Type of feedback
            rating: Star rating (1-5) if applicable
            comment: User comment
            language: Language of the interaction
            user_agent: User's browser/device info
            metadata: Additional metadata (model, tokens, etc.)
        
        Returns:
            Feedback ID
        """
        # Generate feedback ID
        feedback_id = f"fb_{int(time.time() * 1000)}"
        
        # Validate rating
        if rating is not None and (rating < 1 or rating > 5):
            raise ValueError("Rating must be between 1 and 5")
        
        # Create feedback entry
        feedback = UserFeedback(
            id=feedback_id,
            session_id=session_id,
            query=query,
            response=response,
            feedback_type=feedback_type.value,
            rating=rating,
            comment=comment,
            language=language,
            timestamp=time.time(),
            user_agent=user_agent,
            metadata=metadata or {}
        )
        
        # Store in Redis (with 90-day retention)
        self.redis.setex(
            f"{self.feedback_key}:{feedback_id}",
            7776000,  # 90 days
            json.dumps(feedback.to_dict())
        )
        
        # Update analytics
        self._update_analytics(feedback)
        
        logger.info(
            f"Feedback submitted: {feedback_type.value} "
            f"(session: {session_id}, lang: {language})"
        )
        
        return feedback_id
    
    def _update_analytics(self, feedback: UserFeedback):
        """Update feedback analytics"""
        date_key = datetime.fromtimestamp(feedback.timestamp).strftime("%Y-%m-%d")
        analytics_key = f"{self.analytics_key}:{date_key}"
        
        # Get existing analytics
        analytics = self.redis.get(analytics_key)
        if analytics:
            analytics = json.loads(analytics)
        else:
            analytics = {
                'total_feedback': 0,
                'by_type': {},
                'by_language': {},
                'ratings': [],
                'positive_count': 0,
                'negative_count': 0
            }
        
        # Update counts
        analytics['total_feedback'] += 1
        
        # By type
        feedback_type = feedback.feedback_type
        analytics['by_type'][feedback_type] = \
            analytics['by_type'].get(feedback_type, 0) + 1
        
        # By language
        analytics['by_language'][feedback.language] = \
            analytics['by_language'].get(feedback.language, 0) + 1
        
        # Ratings
        if feedback.rating:
            analytics['ratings'].append(feedback.rating)
        
        # Positive/negative
        if feedback_type == FeedbackType.THUMBS_UP.value:
            analytics['positive_count'] += 1
        elif feedback_type == FeedbackType.THUMBS_DOWN.value:
            analytics['negative_count'] += 1
        
        # Save analytics (30-day retention)
        self.redis.setex(
            analytics_key,
            2592000,  # 30 days
            json.dumps(analytics)
        )
    
    def get_feedback(self, feedback_id: str) -> Optional[UserFeedback]:
        """Get feedback by ID"""
        data = self.redis.get(f"{self.feedback_key}:{feedback_id}")
        if data:
            return UserFeedback(**json.loads(data))
        return None
    
    def get_recent_feedback(
        self,
        limit: int = 100,
        feedback_type: Optional[FeedbackType] = None,
        language: Optional[str] = None
    ) -> List[UserFeedback]:
        """
        Get recent feedback entries
        
        Args:
            limit: Maximum number of entries
            feedback_type: Filter by type
            language: Filter by language
        
        Returns:
            List of feedback entries
        """
        pattern = f"{self.feedback_key}:*"
        feedback_list = []
        
        for key in self.redis.scan_iter(match=pattern, count=limit):
            data = self.redis.get(key)
            if not data:
                continue
            
            feedback = UserFeedback(**json.loads(data))
            
            # Apply filters
            if feedback_type and feedback.feedback_type != feedback_type.value:
                continue
            if language and feedback.language != language:
                continue
            
            feedback_list.append(feedback)
        
        # Sort by timestamp (newest first)
        feedback_list.sort(key=lambda x: x.timestamp, reverse=True)
        
        return feedback_list[:limit]
    
    def get_analytics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get feedback analytics for a date range
        
        Args:
            start_date: Start date (default: 7 days ago)
            end_date: End date (default: today)
        
        Returns:
            Aggregated analytics
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=7)
        if end_date is None:
            end_date = datetime.now()
        
        aggregated = {
            'date_range': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'total_feedback': 0,
            'by_type': {},
            'by_language': {},
            'ratings': [],
            'positive_count': 0,
            'negative_count': 0,
            'daily_breakdown': []
        }
        
        # Iterate through date range
        current_date = start_date
        while current_date <= end_date:
            date_key = current_date.strftime("%Y-%m-%d")
            analytics_key = f"{self.analytics_key}:{date_key}"
            
            data = self.redis.get(analytics_key)
            if data:
                daily_data = json.loads(data)
                
                # Aggregate totals
                aggregated['total_feedback'] += daily_data['total_feedback']
                aggregated['positive_count'] += daily_data['positive_count']
                aggregated['negative_count'] += daily_data['negative_count']
                
                # Aggregate by type
                for feedback_type, count in daily_data['by_type'].items():
                    aggregated['by_type'][feedback_type] = \
                        aggregated['by_type'].get(feedback_type, 0) + count
                
                # Aggregate by language
                for lang, count in daily_data['by_language'].items():
                    aggregated['by_language'][lang] = \
                        aggregated['by_language'].get(lang, 0) + count
                
                # Aggregate ratings
                aggregated['ratings'].extend(daily_data['ratings'])
                
                # Add to daily breakdown
                aggregated['daily_breakdown'].append({
                    'date': date_key,
                    **daily_data
                })
            
            current_date += timedelta(days=1)
        
        # Calculate averages
        if aggregated['ratings']:
            aggregated['avg_rating'] = sum(aggregated['ratings']) / len(aggregated['ratings'])
        else:
            aggregated['avg_rating'] = 0.0
        
        total = aggregated['positive_count'] + aggregated['negative_count']
        if total > 0:
            aggregated['satisfaction_rate'] = aggregated['positive_count'] / total * 100
        else:
            aggregated['satisfaction_rate'] = 0.0
        
        return aggregated
    
    def get_negative_feedback_for_improvement(
        self,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get negative feedback for analysis and improvement
        
        Returns:
            List of negative feedback with queries and responses
        """
        negative_feedback = self.get_recent_feedback(
            limit=limit,
            feedback_type=FeedbackType.THUMBS_DOWN
        )
        
        # Format for analysis
        improvement_data = []
        for fb in negative_feedback:
            improvement_data.append({
                'query': fb.query,
                'response': fb.response,
                'comment': fb.comment,
                'language': fb.language,
                'timestamp': datetime.fromtimestamp(fb.timestamp).isoformat(),
                'metadata': fb.metadata
            })
        
        return improvement_data
    
    def export_feedback(
        self,
        format: str = 'json',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> str:
        """
        Export feedback data
        
        Args:
            format: Export format ('json' or 'csv')
            start_date: Start date
            end_date: End date
        
        Returns:
            Exported data as string
        """
        feedback_list = self.get_recent_feedback(limit=10000)
        
        if format == 'json':
            return json.dumps([fb.to_dict() for fb in feedback_list], indent=2)
        elif format == 'csv':
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.DictWriter(
                output,
                fieldnames=[
                    'id', 'session_id', 'query', 'response', 
                    'feedback_type', 'rating', 'comment', 'language', 'timestamp'
                ]
            )
            writer.writeheader()
            
            for fb in feedback_list:
                row = fb.to_dict()
                row.pop('metadata', None)  # Remove complex field
                writer.writerow(row)
            
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format}")


# Example usage
def example_usage():
    """Example of how to use the feedback system"""
    import redis as redis_sync
    
    # Initialize Redis
    redis_client = redis_sync.from_url("redis://localhost:6379")
    
    # Create feedback collector
    collector = FeedbackCollector(redis_client)
    
    # Submit positive feedback
    feedback_id = collector.submit_feedback(
        session_id="sess_12345",
        query="Best restaurants in Taksim?",
        response="Here are some great restaurants in Taksim...",
        feedback_type=FeedbackType.THUMBS_UP,
        language="en",
        metadata={
            'model': 'llama-3.1-8b',
            'tokens': 256,
            'response_time': 2.5
        }
    )
    
    print(f"Feedback submitted: {feedback_id}")
    
    # Submit star rating
    collector.submit_feedback(
        session_id="sess_67890",
        query="How to get to Hagia Sophia?",
        response="Take the tram to Sultanahmet...",
        feedback_type=FeedbackType.STAR_RATING,
        rating=5,
        comment="Very helpful!",
        language="en"
    )
    
    # Get analytics
    analytics = collector.get_analytics()
    print(f"Analytics: {json.dumps(analytics, indent=2)}")
    
    # Get negative feedback for improvement
    negative = collector.get_negative_feedback_for_improvement(limit=10)
    print(f"Negative feedback count: {len(negative)}")


if __name__ == "__main__":
    example_usage()
