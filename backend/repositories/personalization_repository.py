"""
Database repository for personalization and feedback systems
Handles all database operations for persistence
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import desc, func

from backend.models.personalization_models import (
    UserPreference, UserInteraction, UserFeedback, 
    ABTestVariant, ABTestResult, CollaborativeFilteringInteraction
)

logger = logging.getLogger(__name__)


class PersonalizationRepository:
    """Repository for personalization data persistence"""
    
    def __init__(self, db: Session):
        """Initialize repository with database session"""
        self.db = db
    
    # ========== User Preferences ==========
    
    def save_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> UserPreference:
        """Save or update user preferences"""
        user_pref = self.db.query(UserPreference).filter(
            UserPreference.user_id == user_id
        ).first()
        
        if user_pref:
            # Update existing
            user_pref.cuisines = preferences.get('cuisines', {})
            user_pref.price_ranges = preferences.get('price_ranges', {})
            user_pref.districts = preferences.get('districts', {})
            user_pref.activity_types = preferences.get('activity_types', {})
            user_pref.attraction_types = preferences.get('attraction_types', {})
            user_pref.transportation_modes = preferences.get('transportation_modes', {})
            user_pref.time_of_day = preferences.get('time_of_day', {})
            user_pref.interaction_count = preferences.get('interaction_count', 0)
        else:
            # Create new
            user_pref = UserPreference(
                user_id=user_id,
                cuisines=preferences.get('cuisines', {}),
                price_ranges=preferences.get('price_ranges', {}),
                districts=preferences.get('districts', {}),
                activity_types=preferences.get('activity_types', {}),
                attraction_types=preferences.get('attraction_types', {}),
                transportation_modes=preferences.get('transportation_modes', {}),
                time_of_day=preferences.get('time_of_day', {}),
                interaction_count=preferences.get('interaction_count', 0)
            )
            self.db.add(user_pref)
        
        self.db.commit()
        self.db.refresh(user_pref)
        return user_pref
    
    def get_user_preferences(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user preferences from database"""
        user_pref = self.db.query(UserPreference).filter(
            UserPreference.user_id == user_id
        ).first()
        
        if not user_pref:
            return None
        
        return {
            'cuisines': user_pref.cuisines or {},
            'price_ranges': user_pref.price_ranges or {},
            'districts': user_pref.districts or {},
            'activity_types': user_pref.activity_types or {},
            'attraction_types': user_pref.attraction_types or {},
            'transportation_modes': user_pref.transportation_modes or {},
            'time_of_day': user_pref.time_of_day or {},
            'interaction_count': user_pref.interaction_count,
            'last_updated': user_pref.updated_at.isoformat() if user_pref.updated_at else None
        }
    
    # ========== User Interactions ==========
    
    def save_interaction(self, user_id: str, interaction: Dict[str, Any]) -> UserInteraction:
        """Save user interaction"""
        db_interaction = UserInteraction(
            user_id=user_id,
            interaction_id=interaction.get('interaction_id', f"{user_id}_{datetime.now().timestamp()}"),
            interaction_type=interaction.get('type', 'unknown'),
            item_id=interaction.get('item_id'),
            item_data=interaction.get('item_data', {}),
            rating=interaction.get('rating', 0.5)
        )
        
        self.db.add(db_interaction)
        self.db.commit()
        self.db.refresh(db_interaction)
        return db_interaction
    
    def get_user_interactions(self, user_id: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get user interaction history"""
        interactions = self.db.query(UserInteraction).filter(
            UserInteraction.user_id == user_id
        ).order_by(desc(UserInteraction.timestamp)).limit(limit).all()
        
        return [
            {
                'interaction_id': i.interaction_id,
                'type': i.interaction_type,
                'item_id': i.item_id,
                'item_data': i.item_data,
                'rating': i.rating,
                'timestamp': i.timestamp.isoformat() if i.timestamp else None
            }
            for i in interactions
        ]
    
    def get_recent_interactions(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get recent interactions across all users"""
        cutoff = datetime.now() - timedelta(days=days)
        interactions = self.db.query(UserInteraction).filter(
            UserInteraction.timestamp >= cutoff
        ).order_by(desc(UserInteraction.timestamp)).all()
        
        return [
            {
                'user_id': i.user_id,
                'interaction_id': i.interaction_id,
                'type': i.interaction_type,
                'item_id': i.item_id,
                'item_data': i.item_data,
                'rating': i.rating,
                'timestamp': i.timestamp.isoformat() if i.timestamp else None
            }
            for i in interactions
        ]
    
    # ========== User Feedback ==========
    
    def save_feedback(self, user_id: str, interaction_id: str, feedback: Dict[str, Any]) -> UserFeedback:
        """Save user feedback"""
        db_feedback = UserFeedback(
            user_id=user_id,
            interaction_id=interaction_id,
            satisfaction_score=feedback.get('satisfaction_score', 3.0),
            was_helpful=feedback.get('was_helpful', True),
            response_quality=feedback.get('response_quality', 3.0),
            speed_rating=feedback.get('speed_rating', 3.0),
            intent=feedback.get('intent', 'unknown'),
            feature=feedback.get('feature', 'general'),
            comments=feedback.get('comments', ''),
            issues=feedback.get('issues', [])
        )
        
        self.db.add(db_feedback)
        self.db.commit()
        self.db.refresh(db_feedback)
        return db_feedback
    
    def get_user_feedback_history(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get user feedback history"""
        feedback_list = self.db.query(UserFeedback).filter(
            UserFeedback.user_id == user_id
        ).order_by(desc(UserFeedback.created_at)).limit(limit).all()
        
        return [
            {
                'interaction_id': f.interaction_id,
                'satisfaction_score': f.satisfaction_score,
                'was_helpful': f.was_helpful,
                'response_quality': f.response_quality,
                'speed_rating': f.speed_rating,
                'intent': f.intent,
                'feature': f.feature,
                'comments': f.comments,
                'issues': f.issues,
                'timestamp': f.created_at.isoformat() if f.created_at else None
            }
            for f in feedback_list
        ]
    
    def get_aggregate_feedback_metrics(self) -> Dict[str, Any]:
        """Get aggregate feedback metrics"""
        # Total ratings
        total_ratings = self.db.query(func.count(UserFeedback.id)).scalar() or 0
        
        # Average satisfaction
        avg_satisfaction = self.db.query(
            func.avg(UserFeedback.satisfaction_score)
        ).scalar() or 0.0
        
        # Satisfaction by intent
        intent_stats = self.db.query(
            UserFeedback.intent,
            func.count(UserFeedback.id).label('count'),
            func.avg(UserFeedback.satisfaction_score).label('avg_satisfaction')
        ).group_by(UserFeedback.intent).all()
        
        satisfaction_by_intent = {
            intent: {
                'count': count,
                'avg_satisfaction': float(avg_sat),
                'total_score': float(avg_sat * count)
            }
            for intent, count, avg_sat in intent_stats
        }
        
        # Satisfaction by feature
        feature_stats = self.db.query(
            UserFeedback.feature,
            func.count(UserFeedback.id).label('count'),
            func.avg(UserFeedback.satisfaction_score).label('avg_satisfaction')
        ).group_by(UserFeedback.feature).all()
        
        satisfaction_by_feature = {
            feature: {
                'count': count,
                'avg_satisfaction': float(avg_sat),
                'total_score': float(avg_sat * count)
            }
            for feature, count, avg_sat in feature_stats
        }
        
        return {
            'total_ratings': total_ratings,
            'avg_satisfaction': float(avg_satisfaction),
            'satisfaction_by_intent': satisfaction_by_intent,
            'satisfaction_by_feature': satisfaction_by_feature,
            'last_updated': datetime.now().isoformat()
        }
    
    def get_recent_feedback(self, count: int = 100) -> List[Dict[str, Any]]:
        """Get recent feedback entries"""
        feedback_list = self.db.query(UserFeedback).order_by(
            desc(UserFeedback.created_at)
        ).limit(count).all()
        
        return [
            {
                'user_id': f.user_id,
                'interaction_id': f.interaction_id,
                'satisfaction_score': f.satisfaction_score,
                'was_helpful': f.was_helpful,
                'intent': f.intent,
                'feature': f.feature,
                'timestamp': f.created_at.isoformat() if f.created_at else None
            }
            for f in feedback_list
        ]
    
    # ========== A/B Testing ==========
    
    def assign_ab_test_variant(self, test_name: str, user_id: str, variant: str, 
                                experiment_data: Dict[str, Any] = None) -> ABTestVariant:
        """Assign user to A/B test variant"""
        existing = self.db.query(ABTestVariant).filter(
            ABTestVariant.test_name == test_name,
            ABTestVariant.user_id == user_id
        ).first()
        
        if existing:
            return existing
        
        db_variant = ABTestVariant(
            test_name=test_name,
            user_id=user_id,
            variant=variant,
            experiment_data=experiment_data or {}
        )
        
        self.db.add(db_variant)
        self.db.commit()
        self.db.refresh(db_variant)
        return db_variant
    
    def get_ab_test_variant(self, test_name: str, user_id: str) -> Optional[str]:
        """Get user's A/B test variant"""
        variant = self.db.query(ABTestVariant).filter(
            ABTestVariant.test_name == test_name,
            ABTestVariant.user_id == user_id
        ).first()
        
        return variant.variant if variant else None
    
    def record_ab_test_result(self, test_name: str, user_id: str, variant: str,
                              metric_name: str, metric_value: float,
                              interaction_data: Dict[str, Any] = None) -> ABTestResult:
        """Record A/B test result"""
        db_result = ABTestResult(
            test_name=test_name,
            user_id=user_id,
            variant=variant,
            metric_name=metric_name,
            metric_value=metric_value,
            interaction_data=interaction_data or {}
        )
        
        self.db.add(db_result)
        self.db.commit()
        self.db.refresh(db_result)
        return db_result
    
    def get_ab_test_results(self, test_name: str) -> Dict[str, Any]:
        """Get A/B test results by variant"""
        results = self.db.query(
            ABTestResult.variant,
            ABTestResult.metric_name,
            func.count(ABTestResult.id).label('count'),
            func.avg(ABTestResult.metric_value).label('avg_value')
        ).filter(
            ABTestResult.test_name == test_name
        ).group_by(
            ABTestResult.variant, ABTestResult.metric_name
        ).all()
        
        variant_results = {}
        for variant, metric_name, count, avg_value in results:
            if variant not in variant_results:
                variant_results[variant] = {}
            
            variant_results[variant][metric_name] = {
                'count': count,
                'avg': float(avg_value)
            }
        
        return variant_results
    
    # ========== Collaborative Filtering ==========
    
    def save_cf_interaction(self, user_id: str, item_id: str, item_type: str,
                           interaction_score: float, context_data: Dict[str, Any] = None) -> CollaborativeFilteringInteraction:
        """Save collaborative filtering interaction"""
        existing = self.db.query(CollaborativeFilteringInteraction).filter(
            CollaborativeFilteringInteraction.user_id == user_id,
            CollaborativeFilteringInteraction.item_id == item_id
        ).first()
        
        if existing:
            # Update score
            existing.interaction_score = interaction_score
            existing.context_data = context_data or {}
            self.db.commit()
            self.db.refresh(existing)
            return existing
        
        db_interaction = CollaborativeFilteringInteraction(
            user_id=user_id,
            item_id=item_id,
            item_type=item_type,
            interaction_score=interaction_score,
            context_data=context_data or {}
        )
        
        self.db.add(db_interaction)
        self.db.commit()
        self.db.refresh(db_interaction)
        return db_interaction
    
    def get_user_cf_interactions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's collaborative filtering interactions"""
        interactions = self.db.query(CollaborativeFilteringInteraction).filter(
            CollaborativeFilteringInteraction.user_id == user_id
        ).all()
        
        return [
            {
                'item_id': i.item_id,
                'item_type': i.item_type,
                'interaction_score': i.interaction_score,
                'context_data': i.context_data
            }
            for i in interactions
        ]
    
    def get_similar_users(self, user_id: str, min_common_items: int = 3) -> List[str]:
        """Get users with similar interaction patterns"""
        # Get user's items
        user_items = self.db.query(CollaborativeFilteringInteraction.item_id).filter(
            CollaborativeFilteringInteraction.user_id == user_id
        ).all()
        user_item_ids = [item[0] for item in user_items]
        
        if not user_item_ids:
            return []
        
        # Find users who interacted with similar items
        similar_users = self.db.query(
            CollaborativeFilteringInteraction.user_id,
            func.count(CollaborativeFilteringInteraction.item_id).label('common_items')
        ).filter(
            CollaborativeFilteringInteraction.item_id.in_(user_item_ids),
            CollaborativeFilteringInteraction.user_id != user_id
        ).group_by(
            CollaborativeFilteringInteraction.user_id
        ).having(
            func.count(CollaborativeFilteringInteraction.item_id) >= min_common_items
        ).all()
        
        return [user_id for user_id, _ in similar_users]
