"""
Advanced Personalization System
Implements user preference learning, collaborative filtering, and A/B testing
"""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import random
import hashlib

logger = logging.getLogger(__name__)


class UserPreferenceLearner:
    """
    Learns and tracks user preferences over time
    """
    
    def __init__(self):
        """Initialize the preference learner"""
        self.user_preferences = {}  # user_id -> preferences dict
        self.user_interactions = {}  # user_id -> interaction history
        
        # Preference categories
        self.categories = {
            'cuisine': [],
            'price_range': ['budget', 'moderate', 'upscale', 'luxury'],
            'activity_types': ['cultural', 'adventure', 'relaxation', 'nightlife', 'shopping'],
            'districts': [],
            'attraction_types': ['historical', 'modern', 'nature', 'entertainment'],
            'transportation_modes': ['metro', 'bus', 'tram', 'ferry', 'taxi']
        }
        
        logger.info("✅ UserPreferenceLearner initialized")
    
    def record_interaction(self, user_id: str, interaction: Dict[str, Any]):
        """
        Record a user interaction for preference learning
        
        Args:
            user_id: User identifier
            interaction: Dict with keys: type, item_id, item_data, rating, timestamp
        """
        if user_id not in self.user_interactions:
            self.user_interactions[user_id] = []
        
        interaction['timestamp'] = interaction.get('timestamp', datetime.now().isoformat())
        self.user_interactions[user_id].append(interaction)
        
        # Keep only last 1000 interactions per user
        if len(self.user_interactions[user_id]) > 1000:
            self.user_interactions[user_id] = self.user_interactions[user_id][-1000:]
        
        # Update preferences
        self._update_preferences(user_id)
    
    def _update_preferences(self, user_id: str):
        """Update user preferences based on interaction history"""
        if user_id not in self.user_interactions:
            return
        
        interactions = self.user_interactions[user_id]
        preferences = {
            'cuisines': Counter(),
            'price_ranges': Counter(),
            'districts': Counter(),
            'activity_types': Counter(),
            'attraction_types': Counter(),
            'transportation_modes': Counter(),
            'time_of_day': Counter(),
            'interaction_count': len(interactions),
            'last_updated': datetime.now().isoformat()
        }
        
        # Analyze interactions with recency weighting
        now = datetime.now()
        for interaction in interactions:
            # Calculate recency weight (recent interactions weighted higher)
            timestamp = datetime.fromisoformat(interaction['timestamp'])
            days_ago = (now - timestamp).days
            weight = max(1.0, 2.0 - (days_ago / 30))  # Decay over 30 days
            
            # Extract preference signals
            item_data = interaction.get('item_data', {})
            rating = interaction.get('rating', 0.5)  # 0-1 scale
            
            # Only learn from positive interactions (rating > 0.6)
            if rating > 0.6:
                # Cuisine preferences
                if 'cuisine' in item_data:
                    preferences['cuisines'][item_data['cuisine']] += weight
                
                # Price range preferences
                if 'price_range' in item_data:
                    preferences['price_ranges'][item_data['price_range']] += weight
                
                # District preferences
                if 'district' in item_data:
                    preferences['districts'][item_data['district']] += weight
                
                # Activity type preferences
                if 'activity_type' in item_data:
                    preferences['activity_types'][item_data['activity_type']] += weight
                
                # Attraction type preferences
                if 'attraction_type' in item_data:
                    preferences['attraction_types'][item_data['attraction_type']] += weight
                
                # Transportation mode preferences
                if 'transportation_mode' in item_data:
                    preferences['transportation_modes'][item_data['transportation_mode']] += weight
                
                # Time of day preferences
                if 'time_of_day' in item_data:
                    preferences['time_of_day'][item_data['time_of_day']] += weight
        
        # Normalize counters to percentages
        for key in ['cuisines', 'price_ranges', 'districts', 'activity_types', 
                    'attraction_types', 'transportation_modes', 'time_of_day']:
            total = sum(preferences[key].values())
            if total > 0:
                preferences[key] = {k: v/total for k, v in preferences[key].items()}
        
        self.user_preferences[user_id] = preferences
        logger.debug(f"Updated preferences for user {user_id}")
    
    def get_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get learned preferences for a user"""
        return self.user_preferences.get(user_id, {
            'cuisines': {},
            'price_ranges': {},
            'districts': {},
            'activity_types': {},
            'attraction_types': {},
            'transportation_modes': {},
            'time_of_day': {},
            'interaction_count': 0
        })
    
    def get_preference_score(self, user_id: str, item_data: Dict[str, Any]) -> float:
        """
        Calculate preference match score for an item
        
        Returns:
            Score from 0.0 to 1.0
        """
        preferences = self.get_preferences(user_id)
        
        if preferences['interaction_count'] == 0:
            return 0.5  # Neutral for new users
        
        scores = []
        weights = []
        
        # Check cuisine match
        if 'cuisine' in item_data and item_data['cuisine'] in preferences['cuisines']:
            scores.append(preferences['cuisines'][item_data['cuisine']])
            weights.append(0.25)
        
        # Check price range match
        if 'price_range' in item_data and item_data['price_range'] in preferences['price_ranges']:
            scores.append(preferences['price_ranges'][item_data['price_range']])
            weights.append(0.2)
        
        # Check district match
        if 'district' in item_data and item_data['district'] in preferences['districts']:
            scores.append(preferences['districts'][item_data['district']])
            weights.append(0.15)
        
        # Check activity type match
        if 'activity_type' in item_data and item_data['activity_type'] in preferences['activity_types']:
            scores.append(preferences['activity_types'][item_data['activity_type']])
            weights.append(0.2)
        
        # Check attraction type match
        if 'attraction_type' in item_data and item_data['attraction_type'] in preferences['attraction_types']:
            scores.append(preferences['attraction_types'][item_data['attraction_type']])
            weights.append(0.2)
        
        if not scores:
            return 0.5  # Neutral if no matches
        
        # Weighted average
        total_weight = sum(weights)
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        
        return min(1.0, max(0.0, weighted_score))


class CollaborativeFilteringEngine:
    """
    Implements collaborative filtering for recommendations
    """
    
    def __init__(self):
        """Initialize collaborative filtering engine"""
        self.user_item_matrix = {}  # user_id -> {item_id: rating}
        self.item_user_matrix = {}  # item_id -> {user_id: rating}
        self.similarity_cache = {}  # Cache for user similarity scores
        
        logger.info("✅ CollaborativeFilteringEngine initialized")
    
    def record_rating(self, user_id: str, item_id: str, rating: float):
        """Record a user rating for an item"""
        # Update user-item matrix
        if user_id not in self.user_item_matrix:
            self.user_item_matrix[user_id] = {}
        self.user_item_matrix[user_id][item_id] = rating
        
        # Update item-user matrix
        if item_id not in self.item_user_matrix:
            self.item_user_matrix[item_id] = {}
        self.item_user_matrix[item_id][user_id] = rating
        
        # Clear similarity cache for this user
        if user_id in self.similarity_cache:
            del self.similarity_cache[user_id]
    
    def _calculate_user_similarity(self, user1_id: str, user2_id: str) -> float:
        """
        Calculate similarity between two users using cosine similarity
        
        Returns:
            Similarity score from 0.0 to 1.0
        """
        if user1_id not in self.user_item_matrix or user2_id not in self.user_item_matrix:
            return 0.0
        
        user1_ratings = self.user_item_matrix[user1_id]
        user2_ratings = self.user_item_matrix[user2_id]
        
        # Find common items
        common_items = set(user1_ratings.keys()) & set(user2_ratings.keys())
        
        if len(common_items) == 0:
            return 0.0
        
        # Calculate cosine similarity
        numerator = sum(user1_ratings[item] * user2_ratings[item] for item in common_items)
        
        user1_magnitude = sum(user1_ratings[item]**2 for item in common_items)**0.5
        user2_magnitude = sum(user2_ratings[item]**2 for item in common_items)**0.5
        
        if user1_magnitude == 0 or user2_magnitude == 0:
            return 0.0
        
        similarity = numerator / (user1_magnitude * user2_magnitude)
        return max(0.0, min(1.0, similarity))
    
    def get_similar_users(self, user_id: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Find users similar to the given user
        
        Returns:
            List of (user_id, similarity_score) tuples
        """
        if user_id in self.similarity_cache:
            return self.similarity_cache[user_id][:top_n]
        
        similarities = []
        for other_user_id in self.user_item_matrix.keys():
            if other_user_id != user_id:
                similarity = self._calculate_user_similarity(user_id, other_user_id)
                if similarity > 0.1:  # Only consider meaningful similarities
                    similarities.append((other_user_id, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Cache results
        self.similarity_cache[user_id] = similarities
        
        return similarities[:top_n]
    
    def predict_rating(self, user_id: str, item_id: str) -> float:
        """
        Predict a user's rating for an item based on similar users
        
        Returns:
            Predicted rating from 0.0 to 1.0
        """
        # If user has rated this item, return actual rating
        if user_id in self.user_item_matrix and item_id in self.user_item_matrix[user_id]:
            return self.user_item_matrix[user_id][item_id]
        
        # Find similar users who have rated this item
        similar_users = self.get_similar_users(user_id, top_n=20)
        
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for similar_user_id, similarity in similar_users:
            if similar_user_id in self.user_item_matrix and item_id in self.user_item_matrix[similar_user_id]:
                rating = self.user_item_matrix[similar_user_id][item_id]
                weighted_sum += similarity * rating
                weight_sum += similarity
        
        if weight_sum == 0:
            return 0.5  # Neutral prediction if no similar users
        
        predicted_rating = weighted_sum / weight_sum
        return max(0.0, min(1.0, predicted_rating))
    
    def get_collaborative_recommendations(self, user_id: str, candidates: List[str], top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Get recommendations based on collaborative filtering
        
        Args:
            user_id: User to recommend for
            candidates: List of candidate item IDs
            top_n: Number of recommendations to return
        
        Returns:
            List of (item_id, predicted_rating) tuples
        """
        recommendations = []
        
        for item_id in candidates:
            predicted_rating = self.predict_rating(user_id, item_id)
            recommendations.append((item_id, predicted_rating))
        
        # Sort by predicted rating (descending)
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:top_n]


class ABTestingFramework:
    """
    A/B testing framework for response formats and features
    """
    
    def __init__(self):
        """Initialize A/B testing framework"""
        self.experiments = {}  # experiment_id -> experiment config
        self.user_assignments = {}  # user_id -> {experiment_id: variant}
        self.experiment_results = {}  # experiment_id -> {variant: metrics}
        
        logger.info("✅ ABTestingFramework initialized")
    
    def create_experiment(self, experiment_id: str, variants: List[str], 
                         traffic_split: Optional[Dict[str, float]] = None):
        """
        Create a new A/B test experiment
        
        Args:
            experiment_id: Unique identifier for experiment
            variants: List of variant names (e.g., ['control', 'variant_a', 'variant_b'])
            traffic_split: Optional dict of variant -> percentage (0.0-1.0)
        """
        if traffic_split is None:
            # Equal split
            split_percentage = 1.0 / len(variants)
            traffic_split = {v: split_percentage for v in variants}
        
        # Normalize traffic split
        total = sum(traffic_split.values())
        traffic_split = {k: v/total for k, v in traffic_split.items()}
        
        self.experiments[experiment_id] = {
            'variants': variants,
            'traffic_split': traffic_split,
            'created_at': datetime.now().isoformat(),
            'active': True
        }
        
        # Initialize results tracking
        self.experiment_results[experiment_id] = {
            v: {
                'exposures': 0,
                'conversions': 0,
                'satisfaction_scores': [],
                'response_times': []
            } for v in variants
        }
        
        logger.info(f"Created experiment '{experiment_id}' with variants: {variants}")
    
    def assign_variant(self, user_id: str, experiment_id: str) -> str:
        """
        Assign a user to a variant (consistent assignment)
        
        Returns:
            Variant name
        """
        if experiment_id not in self.experiments:
            logger.warning(f"Experiment '{experiment_id}' not found")
            return 'control'
        
        # Check if user already assigned
        if user_id in self.user_assignments and experiment_id in self.user_assignments[user_id]:
            return self.user_assignments[user_id][experiment_id]
        
        # Deterministic assignment based on user_id hash
        experiment = self.experiments[experiment_id]
        hash_value = int(hashlib.md5(f"{user_id}:{experiment_id}".encode()).hexdigest(), 16)
        random_value = (hash_value % 10000) / 10000.0  # 0.0 to 1.0
        
        # Assign to variant based on traffic split
        cumulative = 0.0
        variant = experiment['variants'][0]  # Default to first variant
        
        for v, split in experiment['traffic_split'].items():
            cumulative += split
            if random_value < cumulative:
                variant = v
                break
        
        # Store assignment
        if user_id not in self.user_assignments:
            self.user_assignments[user_id] = {}
        self.user_assignments[user_id][experiment_id] = variant
        
        # Track exposure
        self.experiment_results[experiment_id][variant]['exposures'] += 1
        
        return variant
    
    def record_metric(self, user_id: str, experiment_id: str, metric_name: str, value: float):
        """Record a metric for an experiment"""
        if experiment_id not in self.experiments:
            return
        
        variant = self.user_assignments.get(user_id, {}).get(experiment_id)
        if not variant:
            return
        
        results = self.experiment_results[experiment_id][variant]
        
        if metric_name == 'conversion':
            results['conversions'] += 1
        elif metric_name == 'satisfaction':
            results['satisfaction_scores'].append(value)
        elif metric_name == 'response_time':
            results['response_times'].append(value)
    
    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get results for an experiment"""
        if experiment_id not in self.experiment_results:
            return {}
        
        results_summary = {}
        
        for variant, metrics in self.experiment_results[experiment_id].items():
            exposures = metrics['exposures']
            conversions = metrics['conversions']
            
            results_summary[variant] = {
                'exposures': exposures,
                'conversions': conversions,
                'conversion_rate': conversions / exposures if exposures > 0 else 0.0,
                'avg_satisfaction': sum(metrics['satisfaction_scores']) / len(metrics['satisfaction_scores']) if metrics['satisfaction_scores'] else 0.0,
                'avg_response_time': sum(metrics['response_times']) / len(metrics['response_times']) if metrics['response_times'] else 0.0
            }
        
        return results_summary
    
    def get_winning_variant(self, experiment_id: str, metric: str = 'conversion_rate') -> Tuple[str, float]:
        """
        Determine winning variant based on specified metric
        
        Returns:
            (variant_name, metric_value)
        """
        results = self.get_experiment_results(experiment_id)
        
        if not results:
            return ('control', 0.0)
        
        best_variant = None
        best_value = -float('inf')
        
        for variant, metrics in results.items():
            value = metrics.get(metric, 0.0)
            if value > best_value:
                best_value = value
                best_variant = variant
        
        return (best_variant or 'control', best_value)


class AdvancedPersonalizationSystem:
    """
    Integrated advanced personalization system
    """
    
    def __init__(self):
        """Initialize the advanced personalization system"""
        self.preference_learner = UserPreferenceLearner()
        self.collaborative_filtering = CollaborativeFilteringEngine()
        self.ab_testing = ABTestingFramework()
        
        # Create default experiments
        self._initialize_default_experiments()
        
        logger.info("🎯 Advanced Personalization System initialized")
    
    def _initialize_default_experiments(self):
        """Initialize default A/B test experiments"""
        # Response format experiment
        self.ab_testing.create_experiment(
            'response_format',
            variants=['standard', 'detailed', 'concise'],
            traffic_split={'standard': 0.4, 'detailed': 0.3, 'concise': 0.3}
        )
        
        # Recommendation algorithm experiment
        self.ab_testing.create_experiment(
            'recommendation_algo',
            variants=['ml_only', 'collaborative', 'hybrid'],
            traffic_split={'ml_only': 0.33, 'collaborative': 0.33, 'hybrid': 0.34}
        )
    
    def personalize_recommendations(self, user_id: str, candidates: List[Dict[str, Any]], 
                                   top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Personalize recommendations using learned preferences and collaborative filtering
        
        Args:
            user_id: User identifier
            candidates: List of candidate items with metadata
            top_n: Number of recommendations to return
        
        Returns:
            Personalized list of recommendations
        """
        # Get experiment variant
        variant = self.ab_testing.assign_variant(user_id, 'recommendation_algo')
        
        scored_candidates = []
        
        for candidate in candidates:
            # Get preference-based score
            pref_score = self.preference_learner.get_preference_score(user_id, candidate)
            
            # Get collaborative filtering score
            candidate_id = candidate.get('id', candidate.get('name', ''))
            collab_score = self.collaborative_filtering.predict_rating(user_id, candidate_id)
            
            # Combine scores based on variant
            if variant == 'ml_only':
                final_score = pref_score
            elif variant == 'collaborative':
                final_score = collab_score
            else:  # hybrid
                final_score = 0.6 * pref_score + 0.4 * collab_score
            
            scored_candidates.append((candidate, final_score))
        
        # Sort by score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N with personalization metadata
        recommendations = []
        for candidate, score in scored_candidates[:top_n]:
            candidate['personalization_score'] = score
            candidate['personalization_method'] = variant
            recommendations.append(candidate)
        
        return recommendations
    
    def record_interaction(self, user_id: str, interaction: Dict[str, Any]):
        """Record a user interaction for learning"""
        self.preference_learner.record_interaction(user_id, interaction)
        
        # Also record for collaborative filtering if rating provided
        if 'rating' in interaction and 'item_id' in interaction:
            self.collaborative_filtering.record_rating(
                user_id, 
                interaction['item_id'], 
                interaction['rating']
            )
    
    def get_response_format(self, user_id: str) -> str:
        """Get the response format variant for a user"""
        return self.ab_testing.assign_variant(user_id, 'response_format')
    
    def record_experiment_metric(self, user_id: str, experiment_id: str, 
                                 metric_name: str, value: float):
        """Record an experiment metric"""
        self.ab_testing.record_metric(user_id, experiment_id, metric_name, value)
    
    def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive insights about a user"""
        preferences = self.preference_learner.get_preferences(user_id)
        similar_users = self.collaborative_filtering.get_similar_users(user_id, top_n=5)
        
        return {
            'preferences': preferences,
            'similar_users': [u[0] for u in similar_users],
            'similar_user_count': len(similar_users),
            'personalization_ready': preferences['interaction_count'] >= 5
        }


# Singleton instance
_personalization_system = None


def get_personalization_system() -> AdvancedPersonalizationSystem:
    """Get or create singleton personalization system"""
    global _personalization_system
    if _personalization_system is None:
        _personalization_system = AdvancedPersonalizationSystem()
    return _personalization_system
