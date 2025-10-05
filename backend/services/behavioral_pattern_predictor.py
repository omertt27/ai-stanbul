"""
Behavioral Pattern Prediction System

Advanced system for predicting user behavior patterns and preferences:
- Real-time behavior analysis and prediction
- Dynamic preference learning from user interactions
- Context-aware recommendation adjustment
- Pattern clustering and similarity detection
- Predictive route optimization based on behavioral insights
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BehaviorPattern:
    """Represents a discovered behavior pattern"""
    pattern_id: str
    pattern_type: str  # 'route_preference', 'time_preference', 'cultural_interest', etc.
    confidence_score: float
    frequency: int
    last_observed: datetime
    characteristics: Dict[str, Any]
    user_cluster: Optional[str] = None

@dataclass
class UserBehaviorProfile:
    """Complete behavioral profile for a user"""
    user_id: str
    travel_personality: str  # 'explorer', 'planner', 'spontaneous', 'cultural', 'leisure'
    confidence_level: float
    patterns: List[BehaviorPattern]
    preferences: Dict[str, float]
    interaction_history: List[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime
    prediction_accuracy: float = 0.0

class BehaviorPatternPredictor:
    """Advanced behavioral pattern prediction and learning system"""
    
    def __init__(self, db_path: str = "behavioral_patterns.db"):
        self.db_path = db_path
        self.scaler = StandardScaler()
        self.behavior_clusters = {}
        self.pattern_weights = self._initialize_pattern_weights()
        self.prediction_models = {}
        
        # Initialize database
        self._initialize_database()
        
        # Load existing patterns
        self._load_existing_patterns()
    
    def _initialize_database(self):
        """Initialize SQLite database for behavioral data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # User behavior profiles table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_behavior_profiles (
                    user_id TEXT PRIMARY KEY,
                    travel_personality TEXT,
                    confidence_level REAL,
                    preferences TEXT,  -- JSON
                    prediction_accuracy REAL,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            ''')
            
            # Behavior patterns table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS behavior_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    pattern_type TEXT,
                    confidence_score REAL,
                    frequency INTEGER,
                    last_observed TIMESTAMP,
                    characteristics TEXT,  -- JSON
                    user_cluster TEXT,
                    FOREIGN KEY (user_id) REFERENCES user_behavior_profiles (user_id)
                )
            ''')
            
            # User interactions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_interactions (
                    interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    interaction_type TEXT,
                    interaction_data TEXT,  -- JSON
                    timestamp TIMESTAMP,
                    context TEXT,  -- JSON
                    FOREIGN KEY (user_id) REFERENCES user_behavior_profiles (user_id)
                )
            ''')
            
            # Prediction results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS prediction_results (
                    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    prediction_type TEXT,
                    predicted_value TEXT,  -- JSON
                    actual_value TEXT,     -- JSON
                    accuracy_score REAL,
                    timestamp TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES user_behavior_profiles (user_id)
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
    
    def _initialize_pattern_weights(self) -> Dict[str, float]:
        """Initialize weights for different behavior pattern types"""
        return {
            'route_preference': 0.25,
            'time_preference': 0.20,
            'cultural_interest': 0.15,
            'photo_behavior': 0.10,
            'budget_behavior': 0.10,
            'social_behavior': 0.10,
            'transport_preference': 0.10
        }
    
    def _load_existing_patterns(self):
        """Load existing behavioral patterns from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM user_behavior_profiles')
            user_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM behavior_patterns')
            pattern_count = cursor.fetchone()[0]
            
            conn.close()
            
            logger.info(f"Loaded {user_count} user profiles and {pattern_count} behavior patterns")
            
        except Exception as e:
            logger.error(f"Error loading existing patterns: {e}")
    
    def analyze_user_behavior(self, user_id: str, interaction_data: Dict[str, Any]) -> UserBehaviorProfile:
        """Analyze user behavior and update behavioral profile"""
        
        # Record the interaction
        self._record_interaction(user_id, interaction_data)
        
        # Get or create user profile
        profile = self._get_or_create_user_profile(user_id)
        
        # Extract behavioral patterns from interaction
        new_patterns = self._extract_patterns_from_interaction(user_id, interaction_data)
        
        # Update existing patterns or add new ones
        self._update_behavior_patterns(profile, new_patterns)
        
        # Recalculate travel personality
        profile.travel_personality = self._determine_travel_personality(profile)
        
        # Update confidence level
        profile.confidence_level = self._calculate_confidence_level(profile)
        
        # Save updated profile
        self._save_user_profile(profile)
        
        return profile
    
    def predict_user_preferences(self, user_id: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Predict user preferences based on behavioral patterns"""
        
        profile = self._get_user_profile(user_id)
        if not profile:
            return self._get_default_predictions()
        
        # Get similar users for collaborative filtering
        similar_users = self._find_similar_users(profile)
        
        # Base predictions on user's own patterns
        predictions = self._predict_from_patterns(profile, context)
        
        # Enhance with collaborative filtering
        if similar_users:
            collaborative_predictions = self._collaborative_filtering_predictions(similar_users, context)
            predictions = self._merge_predictions(predictions, collaborative_predictions)
        
        # Apply contextual adjustments
        if context:
            predictions = self._apply_contextual_adjustments(predictions, context)
        
        # Record prediction for later accuracy assessment
        self._record_prediction(user_id, 'preference_prediction', predictions, context)
        
        return predictions
    
    def predict_route_satisfaction(self, user_id: str, route_data: Dict[str, Any]) -> float:
        """Predict how satisfied a user will be with a proposed route"""
        
        profile = self._get_user_profile(user_id)
        if not profile:
            return 0.5  # Neutral prediction for unknown users
        
        satisfaction_score = 0.0
        factor_count = 0
        
        # Analyze route against user patterns
        for pattern in profile.patterns:
            if pattern.pattern_type == 'route_preference':
                alignment = self._calculate_route_pattern_alignment(route_data, pattern)
                satisfaction_score += alignment * pattern.confidence_score
                factor_count += 1
            
            elif pattern.pattern_type == 'time_preference':
                time_alignment = self._calculate_time_preference_alignment(route_data, pattern)
                satisfaction_score += time_alignment * pattern.confidence_score
                factor_count += 1
            
            elif pattern.pattern_type == 'cultural_interest':
                cultural_alignment = self._calculate_cultural_alignment(route_data, pattern)
                satisfaction_score += cultural_alignment * pattern.confidence_score
                factor_count += 1
        
        # Normalize satisfaction score
        if factor_count > 0:
            satisfaction_score = satisfaction_score / factor_count
        else:
            satisfaction_score = 0.5
        
        # Apply travel personality modifiers
        satisfaction_score = self._apply_personality_modifiers(satisfaction_score, profile.travel_personality, route_data)
        
        # Ensure score is within bounds
        satisfaction_score = max(0.0, min(1.0, satisfaction_score))
        
        return satisfaction_score
    
    def learn_from_feedback(self, user_id: str, route_data: Dict[str, Any], 
                          feedback: Dict[str, Any], actual_satisfaction: float):
        """Learn from user feedback to improve future predictions"""
        
        # Get the prediction we made
        predicted_satisfaction = self.predict_route_satisfaction(user_id, route_data)
        
        # Calculate prediction accuracy
        accuracy = 1.0 - abs(predicted_satisfaction - actual_satisfaction)
        
        # Record the feedback
        self._record_prediction(user_id, 'satisfaction_prediction', 
                              {'predicted': predicted_satisfaction, 'actual': actual_satisfaction},
                              {'route_data': route_data, 'feedback': feedback})
        
        # Update user profile based on feedback
        profile = self._get_user_profile(user_id)
        if profile:
            # Update prediction accuracy
            profile.prediction_accuracy = self._update_prediction_accuracy(profile, accuracy)
            
            # Extract new patterns from feedback
            feedback_patterns = self._extract_patterns_from_feedback(user_id, route_data, feedback, actual_satisfaction)
            
            # Update behavior patterns
            self._update_behavior_patterns(profile, feedback_patterns)
            
            # Save updated profile
            self._save_user_profile(profile)
            
            logger.info(f"Updated behavioral patterns for user {user_id} based on feedback")
    
    def cluster_user_behaviors(self, min_interactions: int = 5) -> Dict[str, List[str]]:
        """Cluster users based on behavioral patterns"""
        
        # Get users with sufficient interaction history
        users_data = self._get_users_for_clustering(min_interactions)
        
        if len(users_data) < 3:
            logger.warning("Insufficient data for clustering")
            return {}
        
        # Extract feature vectors
        feature_vectors = []
        user_ids = []
        
        for user_id, user_data in users_data.items():
            features = self._extract_clustering_features(user_data)
            feature_vectors.append(features)
            user_ids.append(user_id)
        
        # Perform clustering
        feature_matrix = np.array(feature_vectors)
        feature_matrix = self.scaler.fit_transform(feature_matrix)
        
        # Determine optimal number of clusters
        n_clusters = min(max(2, len(users_data) // 3), 5)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(feature_matrix)
        
        # Organize results
        clusters = defaultdict(list)
        for user_id, cluster_label in zip(user_ids, cluster_labels):
            clusters[f"cluster_{cluster_label}"].append(user_id)
        
        # Update user profiles with cluster information
        self._update_user_clusters(dict(clusters))
        
        return dict(clusters)
    
    def get_behavioral_insights(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive behavioral insights for a user"""
        
        profile = self._get_user_profile(user_id)
        if not profile:
            return {'error': 'User profile not found'}
        
        insights = {
            'user_id': user_id,
            'travel_personality': profile.travel_personality,
            'confidence_level': profile.confidence_level,
            'prediction_accuracy': profile.prediction_accuracy,
            'total_patterns': len(profile.patterns),
            'pattern_breakdown': self._get_pattern_breakdown(profile),
            'preference_scores': profile.preferences,
            'behavioral_trends': self._analyze_behavioral_trends(profile),
            'recommendations': self._generate_behavioral_recommendations(profile)
        }
        
        return insights
    
    # Private helper methods
    def _record_interaction(self, user_id: str, interaction_data: Dict[str, Any]):
        """Record user interaction in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO user_interactions (user_id, interaction_type, interaction_data, timestamp, context)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                user_id,
                interaction_data.get('type', 'unknown'),
                json.dumps(interaction_data),
                datetime.now(),
                json.dumps(interaction_data.get('context', {}))
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error recording interaction: {e}")
    
    def _get_or_create_user_profile(self, user_id: str) -> UserBehaviorProfile:
        """Get existing user profile or create new one"""
        profile = self._get_user_profile(user_id)
        
        if not profile:
            profile = UserBehaviorProfile(
                user_id=user_id,
                travel_personality='unknown',
                confidence_level=0.0,
                patterns=[],
                preferences={},
                interaction_history=[],
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        
        return profile
    
    def _get_user_profile(self, user_id: str) -> Optional[UserBehaviorProfile]:
        """Get user profile from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT user_id, travel_personality, confidence_level, preferences, 
                       prediction_accuracy, created_at, updated_at
                FROM user_behavior_profiles WHERE user_id = ?
            ''', (user_id,))
            
            row = cursor.fetchone()
            if not row:
                conn.close()
                return None
            
            # Get patterns
            cursor.execute('''
                SELECT pattern_id, pattern_type, confidence_score, frequency,
                       last_observed, characteristics, user_cluster
                FROM behavior_patterns WHERE user_id = ?
            ''', (user_id,))
            
            pattern_rows = cursor.fetchall()
            patterns = []
            
            for pattern_row in pattern_rows:
                pattern = BehaviorPattern(
                    pattern_id=pattern_row[0],
                    pattern_type=pattern_row[1],
                    confidence_score=pattern_row[2],
                    frequency=pattern_row[3],
                    last_observed=datetime.fromisoformat(pattern_row[4]),
                    characteristics=json.loads(pattern_row[5]),
                    user_cluster=pattern_row[6]
                )
                patterns.append(pattern)
            
            conn.close()
            
            profile = UserBehaviorProfile(
                user_id=row[0],
                travel_personality=row[1],
                confidence_level=row[2],
                patterns=patterns,
                preferences=json.loads(row[3]) if row[3] else {},
                interaction_history=[],  # Load separately if needed
                created_at=datetime.fromisoformat(row[5]),
                updated_at=datetime.fromisoformat(row[6]),
                prediction_accuracy=row[4] or 0.0
            )
            
            return profile
            
        except Exception as e:
            logger.error(f"Error getting user profile: {e}")
            return None
    
    def _save_user_profile(self, profile: UserBehaviorProfile):
        """Save user profile to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Update or insert user profile
            cursor.execute('''
                INSERT OR REPLACE INTO user_behavior_profiles 
                (user_id, travel_personality, confidence_level, preferences, 
                 prediction_accuracy, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                profile.user_id,
                profile.travel_personality,
                profile.confidence_level,
                json.dumps(profile.preferences),
                profile.prediction_accuracy,
                profile.created_at,
                datetime.now()
            ))
            
            # Delete existing patterns
            cursor.execute('DELETE FROM behavior_patterns WHERE user_id = ?', (profile.user_id,))
            
            # Insert updated patterns
            for pattern in profile.patterns:
                cursor.execute('''
                    INSERT INTO behavior_patterns 
                    (pattern_id, user_id, pattern_type, confidence_score, frequency,
                     last_observed, characteristics, user_cluster)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    pattern.pattern_id,
                    profile.user_id,
                    pattern.pattern_type,
                    pattern.confidence_score,
                    pattern.frequency,
                    pattern.last_observed,
                    json.dumps(pattern.characteristics),
                    pattern.user_cluster
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving user profile: {e}")
    
    def _extract_patterns_from_interaction(self, user_id: str, interaction_data: Dict[str, Any]) -> List[BehaviorPattern]:
        """Extract behavioral patterns from user interaction"""
        patterns = []
        
        interaction_type = interaction_data.get('type', 'unknown')
        
        if interaction_type == 'route_selection':
            # Extract route preferences
            route_pattern = self._extract_route_preference_pattern(user_id, interaction_data)
            if route_pattern:
                patterns.append(route_pattern)
        
        elif interaction_type == 'location_visit':
            # Extract location preferences
            location_pattern = self._extract_location_preference_pattern(user_id, interaction_data)
            if location_pattern:
                patterns.append(location_pattern)
        
        elif interaction_type == 'time_selection':
            # Extract time preferences
            time_pattern = self._extract_time_preference_pattern(user_id, interaction_data)
            if time_pattern:
                patterns.append(time_pattern)
        
        return patterns
    
    def _extract_route_preference_pattern(self, user_id: str, interaction_data: Dict[str, Any]) -> Optional[BehaviorPattern]:
        """Extract route preference pattern from interaction"""
        route_info = interaction_data.get('route_info', {})
        
        if not route_info:
            return None
        
        pattern_id = f"{user_id}_route_pref_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        characteristics = {
            'preferred_transport': route_info.get('transport_mode', []),
            'distance_preference': route_info.get('total_distance_km', 0),
            'duration_tolerance': route_info.get('total_duration_minutes', 0),
            'cost_sensitivity': route_info.get('total_cost_tl', 0),
            'route_type': route_info.get('route_type', 'unknown')
        }
        
        pattern = BehaviorPattern(
            pattern_id=pattern_id,
            pattern_type='route_preference',
            confidence_score=0.7,  # Initial confidence
            frequency=1,
            last_observed=datetime.now(),
            characteristics=characteristics
        )
        
        return pattern
    
    def _extract_location_preference_pattern(self, user_id: str, interaction_data: Dict[str, Any]) -> Optional[BehaviorPattern]:
        """Extract location preference pattern from interaction"""
        location_info = interaction_data.get('location_info', {})
        
        if not location_info:
            return None
        
        pattern_id = f"{user_id}_location_pref_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        characteristics = {
            'location_type': location_info.get('category', 'unknown'),
            'cultural_significance': location_info.get('cultural_significance', 0),
            'popularity_preference': location_info.get('popularity_score', 0.5),
            'district_preference': location_info.get('district', 'unknown')
        }
        
        pattern = BehaviorPattern(
            pattern_id=pattern_id,
            pattern_type='cultural_interest',
            confidence_score=0.6,
            frequency=1,
            last_observed=datetime.now(),
            characteristics=characteristics
        )
        
        return pattern
    
    def _extract_time_preference_pattern(self, user_id: str, interaction_data: Dict[str, Any]) -> Optional[BehaviorPattern]:
        """Extract time preference pattern from interaction"""
        time_info = interaction_data.get('time_info', {})
        
        if not time_info:
            return None
        
        pattern_id = f"{user_id}_time_pref_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        characteristics = {
            'preferred_hour': time_info.get('hour', 12),
            'day_of_week': time_info.get('day_of_week', 0),
            'season': time_info.get('season', 'unknown'),
            'duration_preference': time_info.get('duration_minutes', 60)
        }
        
        pattern = BehaviorPattern(
            pattern_id=pattern_id,
            pattern_type='time_preference',
            confidence_score=0.5,
            frequency=1,
            last_observed=datetime.now(),
            characteristics=characteristics
        )
        
        return pattern
    
    def _update_behavior_patterns(self, profile: UserBehaviorProfile, new_patterns: List[BehaviorPattern]):
        """Update existing patterns or add new ones"""
        
        for new_pattern in new_patterns:
            # Look for similar existing patterns
            similar_pattern = self._find_similar_pattern(profile.patterns, new_pattern)
            
            if similar_pattern:
                # Update existing pattern
                similar_pattern.frequency += 1
                similar_pattern.last_observed = datetime.now()
                similar_pattern.confidence_score = min(1.0, similar_pattern.confidence_score + 0.1)
                
                # Merge characteristics
                self._merge_pattern_characteristics(similar_pattern, new_pattern)
            else:
                # Add new pattern
                profile.patterns.append(new_pattern)
        
        # Clean up old or low-confidence patterns
        profile.patterns = self._cleanup_patterns(profile.patterns)
    
    def _find_similar_pattern(self, existing_patterns: List[BehaviorPattern], new_pattern: BehaviorPattern) -> Optional[BehaviorPattern]:
        """Find similar pattern in existing patterns"""
        
        for pattern in existing_patterns:
            if (pattern.pattern_type == new_pattern.pattern_type and 
                self._calculate_pattern_similarity(pattern, new_pattern) > 0.7):
                return pattern
        
        return None
    
    def _calculate_pattern_similarity(self, pattern1: BehaviorPattern, pattern2: BehaviorPattern) -> float:
        """Calculate similarity between two patterns"""
        
        if pattern1.pattern_type != pattern2.pattern_type:
            return 0.0
        
        # Simple similarity based on characteristic overlap
        chars1 = set(pattern1.characteristics.keys())
        chars2 = set(pattern2.characteristics.keys())
        
        if not chars1 or not chars2:
            return 0.0
        
        intersection = len(chars1.intersection(chars2))
        union = len(chars1.union(chars2))
        
        return intersection / union if union > 0 else 0.0
    
    def _merge_pattern_characteristics(self, existing_pattern: BehaviorPattern, new_pattern: BehaviorPattern):
        """Merge characteristics from new pattern into existing pattern"""
        
        for key, value in new_pattern.characteristics.items():
            if key in existing_pattern.characteristics:
                # Average numeric values
                if isinstance(value, (int, float)) and isinstance(existing_pattern.characteristics[key], (int, float)):
                    existing_pattern.characteristics[key] = (existing_pattern.characteristics[key] + value) / 2
            else:
                existing_pattern.characteristics[key] = value
    
    def _cleanup_patterns(self, patterns: List[BehaviorPattern]) -> List[BehaviorPattern]:
        """Clean up old or low-confidence patterns"""
        
        # Remove patterns older than 6 months with low confidence
        cutoff_date = datetime.now() - timedelta(days=180)
        
        cleaned_patterns = []
        for pattern in patterns:
            if pattern.last_observed > cutoff_date or pattern.confidence_score > 0.5:
                cleaned_patterns.append(pattern)
        
        return cleaned_patterns
    
    def _determine_travel_personality(self, profile: UserBehaviorProfile) -> str:
        """Determine user's travel personality based on patterns"""
        
        if not profile.patterns:
            return 'unknown'
        
        personality_scores = defaultdict(float)
        
        for pattern in profile.patterns:
            if pattern.pattern_type == 'cultural_interest':
                avg_cultural = np.mean([
                    pattern.characteristics.get('cultural_significance', 0)
                    for pattern in profile.patterns
                    if pattern.pattern_type == 'cultural_interest'
                ])
                if avg_cultural > 0.7:
                    personality_scores['cultural'] += pattern.confidence_score
            
            elif pattern.pattern_type == 'route_preference':
                route_type = pattern.characteristics.get('route_type', 'unknown')
                if route_type == 'fastest':
                    personality_scores['planner'] += pattern.confidence_score
                elif route_type == 'most_scenic':
                    personality_scores['explorer'] += pattern.confidence_score
            
            elif pattern.pattern_type == 'time_preference':
                duration = pattern.characteristics.get('duration_preference', 60)
                if duration > 120:
                    personality_scores['leisure'] += pattern.confidence_score
                else:
                    personality_scores['spontaneous'] += pattern.confidence_score
        
        # Return personality with highest score
        if personality_scores:
            return max(personality_scores.items(), key=lambda x: x[1])[0]
        
        return 'balanced'
    
    def _calculate_confidence_level(self, profile: UserBehaviorProfile) -> float:
        """Calculate overall confidence level for user profile"""
        
        if not profile.patterns:
            return 0.0
        
        # Base confidence on number of patterns and their individual confidence
        pattern_confidence = np.mean([p.confidence_score for p in profile.patterns])
        pattern_count_factor = min(1.0, len(profile.patterns) / 10)  # Max factor at 10 patterns
        
        overall_confidence = pattern_confidence * pattern_count_factor
        
        # Boost confidence if we have recent patterns
        recent_patterns = [p for p in profile.patterns 
                         if p.last_observed > datetime.now() - timedelta(days=30)]
        
        if recent_patterns:
            recency_boost = min(0.2, len(recent_patterns) / 20)
            overall_confidence += recency_boost
        
        return min(1.0, overall_confidence)
    
    def _predict_from_patterns(self, profile: UserBehaviorProfile, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Predict preferences based on user's behavioral patterns"""
        
        predictions = {
            'cultural_interest': 0.5,
            'photo_enthusiasm': 0.5,
            'local_experience_preference': 0.5,
            'crowd_tolerance': 0.5,
            'walking_preference': 0.5,
            'budget_consciousness': 0.5,
            'time_flexibility': 0.5
        }
        
        for pattern in profile.patterns:
            if pattern.pattern_type == 'cultural_interest':
                cultural_sig = pattern.characteristics.get('cultural_significance', 0.5)
                predictions['cultural_interest'] = max(predictions['cultural_interest'], cultural_sig)
            
            elif pattern.pattern_type == 'route_preference':
                cost_sensitivity = pattern.characteristics.get('cost_sensitivity', 0)
                if cost_sensitivity < 20:
                    predictions['budget_consciousness'] = max(predictions['budget_consciousness'], 0.8)
                
                distance_pref = pattern.characteristics.get('distance_preference', 0)
                if distance_pref < 2:
                    predictions['walking_preference'] = max(predictions['walking_preference'], 0.7)
        
        return predictions
    
    def _find_similar_users(self, profile: UserBehaviorProfile) -> List[str]:
        """Find users with similar behavioral patterns"""
        
        # This would implement collaborative filtering
        # For now, return empty list
        return []
    
    def _collaborative_filtering_predictions(self, similar_users: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate predictions based on similar users"""
        
        # Placeholder for collaborative filtering implementation
        return {}
    
    def _merge_predictions(self, base_predictions: Dict[str, Any], collaborative_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Merge base predictions with collaborative filtering predictions"""
        
        # Simple weighted average for now
        merged = {}
        
        for key in base_predictions.keys():
            base_value = base_predictions[key]
            collab_value = collaborative_predictions.get(key, base_value)
            merged[key] = (base_value * 0.7) + (collab_value * 0.3)
        
        return merged
    
    def _apply_contextual_adjustments(self, predictions: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply contextual adjustments to predictions"""
        
        # Weather adjustments
        weather = context.get('weather', 'unknown')
        if weather == 'rainy':
            predictions['walking_preference'] *= 0.7
        
        # Time of day adjustments
        hour = context.get('hour', 12)
        if hour < 9 or hour > 18:
            predictions['crowd_tolerance'] *= 1.2  # Less crowded times
        
        # Season adjustments
        season = context.get('season', 'unknown')
        if season == 'winter':
            predictions['walking_preference'] *= 0.8
        
        return predictions
    
    def _record_prediction(self, user_id: str, prediction_type: str, prediction_data: Dict[str, Any], context: Dict[str, Any]):
        """Record prediction for later accuracy assessment"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO prediction_results (user_id, prediction_type, predicted_value, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (
                user_id,
                prediction_type,
                json.dumps(prediction_data),
                datetime.now()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error recording prediction: {e}")
    
    def _get_default_predictions(self) -> Dict[str, Any]:
        """Get default predictions for unknown users"""
        
        return {
            'cultural_interest': 0.5,
            'photo_enthusiasm': 0.5,
            'local_experience_preference': 0.5,
            'crowd_tolerance': 0.5,
            'walking_preference': 0.6,
            'budget_consciousness': 0.6,
            'time_flexibility': 0.5,
            'travel_style': 'balanced'
        }
    
    # Additional helper methods would be implemented here...
    
    def _calculate_route_pattern_alignment(self, route_data: Dict[str, Any], pattern: BehaviorPattern) -> float:
        """Calculate alignment between route and route preference pattern"""
        return 0.5  # Placeholder
    
    def _calculate_time_preference_alignment(self, route_data: Dict[str, Any], pattern: BehaviorPattern) -> float:
        """Calculate alignment between route timing and time preference pattern"""
        return 0.5  # Placeholder
    
    def _calculate_cultural_alignment(self, route_data: Dict[str, Any], pattern: BehaviorPattern) -> float:
        """Calculate cultural alignment between route and cultural interest pattern"""
        return 0.5  # Placeholder
    
    def _apply_personality_modifiers(self, base_score: float, personality: str, route_data: Dict[str, Any]) -> float:
        """Apply personality-based modifiers to satisfaction score"""
        
        modifiers = {
            'explorer': 0.1,
            'cultural': 0.05,
            'planner': -0.05,
            'spontaneous': 0.0,
            'leisure': 0.0
        }
        
        return base_score + modifiers.get(personality, 0.0)
    
    def _update_prediction_accuracy(self, profile: UserBehaviorProfile, new_accuracy: float) -> float:
        """Update running prediction accuracy"""
        
        if profile.prediction_accuracy == 0.0:
            return new_accuracy
        
        # Running average with more weight on recent accuracy
        return (profile.prediction_accuracy * 0.8) + (new_accuracy * 0.2)
    
    def _extract_patterns_from_feedback(self, user_id: str, route_data: Dict[str, Any], 
                                      feedback: Dict[str, Any], satisfaction: float) -> List[BehaviorPattern]:
        """Extract patterns from user feedback"""
        
        patterns = []
        
        # Extract satisfaction-based patterns
        if satisfaction > 0.8:
            # User really liked this route - extract positive patterns
            pattern_id = f"{user_id}_positive_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            characteristics = {
                'liked_features': feedback.get('liked_features', []),
                'satisfaction_score': satisfaction,
                'route_characteristics': route_data
            }
            
            pattern = BehaviorPattern(
                pattern_id=pattern_id,
                pattern_type='route_preference',
                confidence_score=0.8,
                frequency=1,
                last_observed=datetime.now(),
                characteristics=characteristics
            )
            
            patterns.append(pattern)
        
        return patterns
    
    def _get_users_for_clustering(self, min_interactions: int) -> Dict[str, Dict[str, Any]]:
        """Get users with sufficient data for clustering"""
        return {}  # Placeholder
    
    def _extract_clustering_features(self, user_data: Dict[str, Any]) -> List[float]:
        """Extract feature vector for clustering"""
        return [0.5] * 10  # Placeholder
    
    def _update_user_clusters(self, clusters: Dict[str, List[str]]):
        """Update user cluster assignments in database"""
        pass  # Placeholder
    
    def _get_pattern_breakdown(self, profile: UserBehaviorProfile) -> Dict[str, int]:
        """Get breakdown of patterns by type"""
        
        breakdown = defaultdict(int)
        for pattern in profile.patterns:
            breakdown[pattern.pattern_type] += 1
        
        return dict(breakdown)
    
    def _analyze_behavioral_trends(self, profile: UserBehaviorProfile) -> Dict[str, Any]:
        """Analyze trends in user behavior"""
        
        trends = {
            'pattern_evolution': 'stable',
            'confidence_trend': 'increasing',
            'recent_changes': []
        }
        
        return trends
    
    def _generate_behavioral_recommendations(self, profile: UserBehaviorProfile) -> List[str]:
        """Generate recommendations based on behavioral analysis"""
        
        recommendations = []
        
        if profile.confidence_level < 0.5:
            recommendations.append("Continue using the system to improve personalization")
        
        if profile.travel_personality == 'cultural':
            recommendations.append("Consider visiting lesser-known historical sites")
        
        return recommendations

# Factory function
def create_behavior_predictor() -> BehaviorPatternPredictor:
    """Create and return behavior pattern predictor instance"""
    return BehaviorPatternPredictor()

# Example usage
if __name__ == "__main__":
    predictor = create_behavior_predictor()
    
    # Example interaction analysis
    interaction_data = {
        'type': 'route_selection',
        'route_info': {
            'transport_mode': ['walking', 'tram'],
            'total_distance_km': 2.5,
            'total_duration_minutes': 35,
            'total_cost_tl': 7.67,
            'route_type': 'cultural_immersion'
        },
        'context': {
            'time': datetime.now(),
            'weather': 'sunny',
            'group_size': 2
        }
    }
    
    # Analyze behavior
    profile = predictor.analyze_user_behavior('test_user_123', interaction_data)
    print(f"Travel personality: {profile.travel_personality}")
    print(f"Confidence level: {profile.confidence_level:.2f}")
    
    # Predict preferences
    predictions = predictor.predict_user_preferences('test_user_123')
    print(f"Predicted preferences: {predictions}")
    
    # Get behavioral insights
    insights = predictor.get_behavioral_insights('test_user_123')
    print(f"Behavioral insights: {insights}")
