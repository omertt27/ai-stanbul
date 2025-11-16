"""
Personalization Module for Pure LLM Handler

This module provides:
1. User profile management
2. Preference learning from interactions
3. Personalized context filtering
4. Feedback loop for continuous improvement

Author: AI Istanbul Team
Date: January 2025
"""

import logging
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
import json

logger = logging.getLogger(__name__)


@dataclass
class UserPreferences:
    """User preference profile"""
    user_id: str
    
    # Cuisine preferences
    preferred_cuisines: List[str] = field(default_factory=list)
    disliked_cuisines: List[str] = field(default_factory=list)
    
    # Price preferences
    preferred_price_range: Optional[str] = None  # budget, moderate, upscale
    
    # Interest categories
    interests: List[str] = field(default_factory=list)  # history, art, food, nature, etc.
    
    # Location preferences
    preferred_districts: List[str] = field(default_factory=list)
    
    # Activity preferences
    preferred_activities: List[str] = field(default_factory=list)  # museums, restaurants, parks, etc.
    
    # Time preferences
    preferred_times: List[str] = field(default_factory=list)  # morning, afternoon, evening
    
    # Interaction history
    query_count: int = 0
    positive_feedback_count: int = 0
    negative_feedback_count: int = 0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_interaction: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        if self.last_interaction:
            data['last_interaction'] = self.last_interaction.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserPreferences':
        """Create from dictionary"""
        # Convert ISO strings back to datetime
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        if 'last_interaction' in data and data['last_interaction']:
            data['last_interaction'] = datetime.fromisoformat(data['last_interaction'])
        return cls(**data)


@dataclass
class FeedbackRecord:
    """Record of user feedback"""
    user_id: str
    query: str
    response: str
    feedback_type: str  # positive, negative, correction
    feedback_details: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Detected signals at time of query
    detected_signals: List[str] = field(default_factory=list)
    signal_scores: Dict[str, float] = field(default_factory=dict)
    
    # What was wrong (for negative feedback)
    issues: List[str] = field(default_factory=list)  # wrong_intent, irrelevant, etc.
    
    # Corrections (for learning)
    correct_signals: Optional[List[str]] = None
    correct_intent: Optional[str] = None


class PersonalizationEngine:
    """
    Personalization engine for learning user preferences and improving responses.
    
    Features:
    - User profile management
    - Preference learning from interactions
    - Feedback processing
    - Personalized context filtering
    """
    
    def __init__(
        self,
        db_connection=None,
        redis_client=None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize personalization engine.
        
        Args:
            db_connection: Database connection for persistent storage
            redis_client: Redis client for caching profiles
            config: Configuration dictionary
        """
        self.db = db_connection
        self.redis = redis_client
        self.config = config or {}
        
        # In-memory cache of user profiles
        self.profiles: Dict[str, UserPreferences] = {}
        
        # Feedback history for learning
        self.feedback_history: List[FeedbackRecord] = []
        
        # Learning configuration
        self.min_interactions_for_personalization = self.config.get(
            'min_interactions', 3
        )
        self.preference_confidence_threshold = self.config.get(
            'preference_threshold', 0.6
        )
        
        logger.info("✅ Personalization Engine initialized")
    
    async def get_user_profile(self, user_id: str) -> UserPreferences:
        """
        Get or create user profile.
        
        Args:
            user_id: User identifier
            
        Returns:
            User preferences profile
        """
        # Check in-memory cache
        if user_id in self.profiles:
            return self.profiles[user_id]
        
        # Check Redis cache
        if self.redis:
            try:
                cached = await self._get_from_redis(user_id)
                if cached:
                    self.profiles[user_id] = cached
                    return cached
            except Exception as e:
                logger.warning(f"Redis cache miss for user {user_id}: {e}")
        
        # Check database
        if self.db:
            try:
                profile = await self._get_from_database(user_id)
                if profile:
                    self.profiles[user_id] = profile
                    return profile
            except Exception as e:
                logger.warning(f"Database lookup failed for user {user_id}: {e}")
        
        # Create new profile
        profile = UserPreferences(user_id=user_id)
        self.profiles[user_id] = profile
        logger.info(f"Created new profile for user {user_id}")
        
        return profile
    
    async def update_profile_from_interaction(
        self,
        user_id: str,
        query: str,
        selected_items: List[Dict[str, Any]],
        signals: List[str]
    ):
        """
        Learn from user interaction with results.
        
        Args:
            user_id: User identifier
            query: User's query
            selected_items: Items user interacted with (clicked, viewed, etc.)
            signals: Detected signals in the query
        """
        profile = await self.get_user_profile(user_id)
        
        # Update interaction count
        profile.query_count += 1
        profile.last_interaction = datetime.now()
        profile.updated_at = datetime.now()
        
        # Learn from selected items
        for item in selected_items:
            item_type = item.get('type', 'unknown')
            
            # Learn cuisine preferences
            if item_type == 'restaurant' and 'cuisine' in item:
                cuisine = item['cuisine']
                if cuisine not in profile.preferred_cuisines:
                    profile.preferred_cuisines.append(cuisine)
            
            # Learn location preferences
            if 'district' in item:
                district = item['district']
                if district not in profile.preferred_districts:
                    profile.preferred_districts.append(district)
            
            # Learn price preferences
            if 'price_level' in item:
                price = item['price_level']
                if not profile.preferred_price_range:
                    profile.preferred_price_range = price
        
        # Learn from signals
        for signal in signals:
            if signal.startswith('needs_'):
                activity = signal.replace('needs_', '')
                if activity not in profile.preferred_activities:
                    profile.preferred_activities.append(activity)
        
        # Save profile
        await self._save_profile(profile)
        
        logger.info(f"Updated profile for user {user_id} from interaction")
    
    async def process_feedback(
        self,
        user_id: str,
        query: str,
        response: str,
        feedback_type: str,
        detected_signals: List[str],
        signal_scores: Dict[str, float],
        feedback_details: Optional[Dict[str, Any]] = None
    ) -> FeedbackRecord:
        """
        Process user feedback to improve the system.
        
        Args:
            user_id: User identifier
            query: Original query
            response: System response
            feedback_type: 'positive', 'negative', or 'correction'
            detected_signals: Signals detected for this query
            signal_scores: Confidence scores for each signal
            feedback_details: Additional feedback information
            
        Returns:
            Feedback record
        """
        profile = await self.get_user_profile(user_id)
        
        # Create feedback record
        feedback = FeedbackRecord(
            user_id=user_id,
            query=query,
            response=response,
            feedback_type=feedback_type,
            detected_signals=detected_signals,
            signal_scores=signal_scores,
            feedback_details=feedback_details or {}
        )
        
        # Update profile based on feedback
        if feedback_type == 'positive':
            profile.positive_feedback_count += 1
            # Reinforce detected signals (they were correct)
            logger.info(f"Positive feedback for signals: {detected_signals}")
            
        elif feedback_type == 'negative':
            profile.negative_feedback_count += 1
            # Extract what was wrong
            if feedback_details:
                feedback.issues = feedback_details.get('issues', [])
                feedback.correct_signals = feedback_details.get('correct_signals')
                feedback.correct_intent = feedback_details.get('correct_intent')
            
            logger.info(f"Negative feedback for query: {query[:50]}...")
        
        # Store feedback
        self.feedback_history.append(feedback)
        
        # Save to database
        if self.db:
            try:
                await self._save_feedback_to_database(feedback)
            except Exception as e:
                logger.error(f"Failed to save feedback to database: {e}")
        
        # Save updated profile
        await self._save_profile(profile)
        
        return feedback
    
    async def filter_context_by_preferences(
        self,
        user_id: str,
        context_items: List[Dict[str, Any]],
        signals: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Filter and rank context items based on user preferences.
        
        Args:
            user_id: User identifier
            context_items: List of context items (restaurants, museums, etc.)
            signals: Detected signals for relevance
            
        Returns:
            Filtered and ranked context items
        """
        profile = await self.get_user_profile(user_id)
        
        # Don't personalize for new users
        if profile.query_count < self.min_interactions_for_personalization:
            return context_items
        
        # Score each item based on preferences
        scored_items = []
        for item in context_items:
            score = self._calculate_preference_score(item, profile, signals)
            scored_items.append((score, item))
        
        # Sort by score (highest first)
        scored_items.sort(key=lambda x: x[0], reverse=True)
        
        # Return top items
        max_items = self.config.get('max_personalized_items', 20)
        filtered = [item for score, item in scored_items[:max_items]]
        
        logger.info(
            f"Filtered {len(context_items)} items to {len(filtered)} "
            f"based on user preferences"
        )
        
        return filtered
    
    def _calculate_preference_score(
        self,
        item: Dict[str, Any],
        profile: UserPreferences,
        signals: List[str]
    ) -> float:
        """Calculate preference score for an item"""
        score = 1.0  # Base score
        
        # Cuisine preference boost
        if 'cuisine' in item and item['cuisine'] in profile.preferred_cuisines:
            score += 0.5
        
        # Disliked cuisine penalty
        if 'cuisine' in item and item['cuisine'] in profile.disliked_cuisines:
            score -= 0.7
        
        # District preference boost
        if 'district' in item and item['district'] in profile.preferred_districts:
            score += 0.3
        
        # Price match boost
        if 'price_level' in item and item['price_level'] == profile.preferred_price_range:
            score += 0.2
        
        # Interest category boost
        if 'category' in item:
            for interest in profile.interests:
                if interest.lower() in item['category'].lower():
                    score += 0.4
        
        return max(score, 0.0)  # Ensure non-negative
    
    async def get_feedback_summary(
        self,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get feedback summary for learning and analysis.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Feedback summary with statistics
        """
        cutoff = datetime.now() - timedelta(days=days)
        recent_feedback = [
            f for f in self.feedback_history
            if f.timestamp >= cutoff
        ]
        
        # Calculate statistics
        total = len(recent_feedback)
        positive = sum(1 for f in recent_feedback if f.feedback_type == 'positive')
        negative = sum(1 for f in recent_feedback if f.feedback_type == 'negative')
        
        # Signal accuracy
        signal_stats = defaultdict(lambda: {'correct': 0, 'incorrect': 0})
        for feedback in recent_feedback:
            if feedback.feedback_type == 'positive':
                for signal in feedback.detected_signals:
                    signal_stats[signal]['correct'] += 1
            elif feedback.feedback_type == 'negative' and feedback.correct_signals:
                for signal in feedback.detected_signals:
                    if signal not in feedback.correct_signals:
                        signal_stats[signal]['incorrect'] += 1
        
        return {
            'period_days': days,
            'total_feedback': total,
            'positive_feedback': positive,
            'negative_feedback': negative,
            'satisfaction_rate': positive / total if total > 0 else 0.0,
            'signal_accuracy': {
                signal: stats['correct'] / (stats['correct'] + stats['incorrect'])
                if (stats['correct'] + stats['incorrect']) > 0 else 0.0
                for signal, stats in signal_stats.items()
            }
        }
    
    async def get_personalization_metrics(self) -> Dict[str, Any]:
        """Get personalization metrics"""
        total_users = len(self.profiles)
        personalized_users = sum(
            1 for p in self.profiles.values()
            if p.query_count >= self.min_interactions_for_personalization
        )
        
        return {
            'total_users': total_users,
            'personalized_users': personalized_users,
            'personalization_rate': personalized_users / total_users if total_users > 0 else 0.0,
            'avg_queries_per_user': sum(p.query_count for p in self.profiles.values()) / total_users if total_users > 0 else 0.0,
            'avg_positive_feedback': sum(p.positive_feedback_count for p in self.profiles.values()) / total_users if total_users > 0 else 0.0,
            'total_feedback_records': len(self.feedback_history)
        }
    
    # Storage methods
    
    async def _save_profile(self, profile: UserPreferences):
        """Save profile to storage"""
        # Update in-memory cache
        self.profiles[profile.user_id] = profile
        
        # Save to Redis
        if self.redis:
            try:
                await self._save_to_redis(profile)
            except Exception as e:
                logger.error(f"Failed to save profile to Redis: {e}")
        
        # Save to database
        if self.db:
            try:
                await self._save_to_database(profile)
            except Exception as e:
                logger.error(f"Failed to save profile to database: {e}")
    
    async def _get_from_redis(self, user_id: str) -> Optional[UserPreferences]:
        """Get profile from Redis"""
        if not self.redis:
            return None
        
        key = f"user_profile:{user_id}"
        data = await self.redis.get(key)
        if data:
            profile_dict = json.loads(data)
            return UserPreferences.from_dict(profile_dict)
        return None
    
    async def _save_to_redis(self, profile: UserPreferences):
        """Save profile to Redis"""
        if not self.redis:
            return
        
        key = f"user_profile:{profile.user_id}"
        data = json.dumps(profile.to_dict())
        await self.redis.set(key, data, ex=86400 * 30)  # 30 days expiry
    
    async def _get_from_database(self, user_id: str) -> Optional[UserPreferences]:
        """Get profile from database"""
        # TODO: Implement database lookup
        return None
    
    async def _save_to_database(self, profile: UserPreferences):
        """Save profile to database"""
        # TODO: Implement database save
        pass
    
    async def _save_feedback_to_database(self, feedback: FeedbackRecord):
        """Save feedback to database"""
        # TODO: Implement feedback save
        pass
