"""
Online Learning System for Real-Time Personalization
Implements incremental learning, Thompson Sampling, and concept drift detection
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import math

# Import numpy with fallback
try:
    import numpy as np
    NUMPY_AVAILABLE = True
    NDArray = np.ndarray
except ImportError:
    NUMPY_AVAILABLE = False
    np = None
    NDArray = List[float]  # Fallback type for when numpy isn't available

# Import scipy with fallback
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None

logger = logging.getLogger(__name__)

# Log once at startup if optional packages are missing (info level, not warning)
if not NUMPY_AVAILABLE:
    logger.info("ℹ️  NumPy not available - using Python fallback implementations")
if not SCIPY_AVAILABLE:
    logger.info("ℹ️  SciPy not available - using simplified statistical functions")


class ThompsonSampling:
    """
    Thompson Sampling for contextual bandits
    Balances exploration vs exploitation for personalized recommendations
    """
    
    def __init__(self, n_arms: int, alpha_prior: float = 1.0, beta_prior: float = 1.0):
        """
        Initialize Thompson Sampling
        
        Args:
            n_arms: Number of arms (recommendation candidates)
            alpha_prior: Prior alpha parameter for Beta distribution
            beta_prior: Prior beta parameter for Beta distribution
        """
        self.n_arms = n_arms
        if NUMPY_AVAILABLE:
            self.alpha = np.ones(n_arms) * alpha_prior
            self.beta = np.ones(n_arms) * beta_prior
            self.total_pulls = np.zeros(n_arms)
            self.total_rewards = np.zeros(n_arms)
        else:
            self.alpha = [alpha_prior] * n_arms
            self.beta = [beta_prior] * n_arms
            self.total_pulls = [0.0] * n_arms
            self.total_rewards = [0.0] * n_arms
        
        logger.info(f"✅ ThompsonSampling initialized with {n_arms} arms")
    
    def select_arm(self) -> int:
        """
        Select an arm using Thompson Sampling
        
        Returns:
            int: Selected arm index
        """
        if NUMPY_AVAILABLE:
            # Sample from Beta distribution for each arm
            samples = np.random.beta(self.alpha, self.beta)
            return int(np.argmax(samples))
        else:
            # Fallback: Simple epsilon-greedy
            import random
            if random.random() < 0.1:  # 10% exploration
                return random.randint(0, self.n_arms - 1)
            else:  # 90% exploitation
                # Select arm with highest mean reward
                means = [self.total_rewards[i] / max(self.total_pulls[i], 1) for i in range(self.n_arms)]
                return means.index(max(means))
    
    def update(self, arm: int, reward: float):
        """
        Update arm statistics based on observed reward
        
        Args:
            arm: Arm index that was selected
            reward: Observed reward (0-1 range recommended)
        """
        self.total_pulls[arm] += 1
        self.total_rewards[arm] += reward
        
        # Update Beta distribution parameters
        self.alpha[arm] += reward
        self.beta[arm] += (1 - reward)
    
    def get_arm_stats(self) -> List[Dict[str, float]]:
        """Get statistics for all arms"""
        stats = []
        for i in range(self.n_arms):
            mean = self.alpha[i] / (self.alpha[i] + self.beta[i])
            var = (self.alpha[i] * self.beta[i]) / \
                  ((self.alpha[i] + self.beta[i])**2 * (self.alpha[i] + self.beta[i] + 1))
            stats.append({
                'arm': i,
                'mean': mean,
                'variance': var,
                'pulls': int(self.total_pulls[i]),
                'total_reward': float(self.total_rewards[i])
            })
        return stats


class ConceptDriftDetector:
    """
    Detects concept drift in user behavior using ADWIN (Adaptive Windowing)
    Triggers model retraining when significant distribution shifts are detected
    """
    
    def __init__(self, window_size: int = 100, delta: float = 0.002):
        """
        Initialize drift detector
        
        Args:
            window_size: Maximum window size for storing recent observations
            delta: Confidence parameter for drift detection (lower = more sensitive)
        """
        self.window = deque(maxlen=window_size)
        self.delta = delta
        self.drift_detected = False
        self.drift_count = 0
        self.total_observations = 0
        
        logger.info(f"✅ ConceptDriftDetector initialized (window_size={window_size}, delta={delta})")
    
    def add_observation(self, value: float) -> bool:
        """
        Add a new observation and check for drift
        
        Args:
            value: Observation value (e.g., prediction error, CTR)
        
        Returns:
            bool: True if drift was detected
        """
        self.window.append(value)
        self.total_observations += 1
        
        # Only check for drift if we have enough observations
        if len(self.window) < 30:
            return False
        
        # Split window into two parts and compare distributions
        mid = len(self.window) // 2
        window_a = list(self.window)[:mid]
        window_b = list(self.window)[mid:]
        
        # Perform two-sample t-test
        if len(window_a) > 1 and len(window_b) > 1:
            t_stat, p_value = stats.ttest_ind(window_a, window_b)
            
            # Drift detected if p-value is below threshold
            if p_value < self.delta:
                self.drift_detected = True
                self.drift_count += 1
                logger.warning(f"🚨 Concept drift detected! (p-value={p_value:.6f})")
                return True
        
        self.drift_detected = False
        return False
    
    def reset(self):
        """Reset the detector"""
        self.window.clear()
        self.drift_detected = False


class IncrementalEmbeddingLearner:
    """
    Incrementally learns user and item embeddings using stochastic gradient descent
    Enables real-time personalization without full model retraining
    """
    
    def __init__(self, embedding_dim: int = 64, learning_rate: float = 0.01, 
                 l2_reg: float = 0.001):
        """
        Initialize embedding learner
        
        Args:
            embedding_dim: Dimension of embedding vectors
            learning_rate: Learning rate for SGD
            l2_reg: L2 regularization parameter
        """
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        
        self.user_embeddings = {}  # user_id -> embedding vector
        self.item_embeddings = {}  # item_id -> embedding vector
        
        self.user_bias = {}  # user_id -> bias term
        self.item_bias = {}  # item_id -> bias term
        self.global_bias = 0.0
        
        self.update_count = 0
        
        logger.info(f"✅ IncrementalEmbeddingLearner initialized (dim={embedding_dim}, lr={learning_rate})")
    
    def _initialize_user(self, user_id: str):
        """Initialize embeddings for a new user"""
        if user_id not in self.user_embeddings:
            if NUMPY_AVAILABLE:
                self.user_embeddings[user_id] = np.random.normal(0, 0.1, self.embedding_dim)
            else:
                import random
                self.user_embeddings[user_id] = [random.gauss(0, 0.1) for _ in range(self.embedding_dim)]
            self.user_bias[user_id] = 0.0
    
    def _initialize_item(self, item_id: str):
        """Initialize embeddings for a new item"""
        if item_id not in self.item_embeddings:
            if NUMPY_AVAILABLE:
                self.item_embeddings[item_id] = np.random.normal(0, 0.1, self.embedding_dim)
            else:
                import random
                self.item_embeddings[item_id] = [random.gauss(0, 0.1) for _ in range(self.embedding_dim)]
            self.item_bias[item_id] = 0.0
    
    def predict(self, user_id: str, item_id: str) -> float:
        """
        Predict user-item interaction score
        
        Args:
            user_id: User identifier
            item_id: Item identifier
        
        Returns:
            float: Predicted score
        """
        self._initialize_user(user_id)
        self._initialize_item(item_id)
        
        user_emb = self.user_embeddings[user_id]
        item_emb = self.item_embeddings[item_id]
        
        # Dot product + biases
        if NUMPY_AVAILABLE:
            score = np.dot(user_emb, item_emb) + \
                    self.user_bias[user_id] + \
                    self.item_bias[item_id] + \
                    self.global_bias
        else:
            # Fallback: manual dot product
            score = sum(a * b for a, b in zip(user_emb, item_emb)) + \
                    self.user_bias[user_id] + \
                    self.item_bias[item_id] + \
                    self.global_bias
        
        # Apply sigmoid to bound between 0 and 1
        return 1.0 / (1.0 + np.exp(-score))
    
    def update(self, user_id: str, item_id: str, rating: float):
        """
        Update embeddings based on observed interaction
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            rating: Observed rating/reward (0-1 range)
        """
        self._initialize_user(user_id)
        self._initialize_item(item_id)
        
        # Get current prediction
        prediction = self.predict(user_id, item_id)
        
        # Compute error
        error = rating - prediction
        
        # Get embeddings
        user_emb = self.user_embeddings[user_id]
        item_emb = self.item_embeddings[item_id]
        
        # Update embeddings with gradient descent
        user_grad = error * item_emb - self.l2_reg * user_emb
        item_grad = error * user_emb - self.l2_reg * item_emb
        
        self.user_embeddings[user_id] += self.learning_rate * user_grad
        self.item_embeddings[item_id] += self.learning_rate * item_grad
        
        # Update biases
        self.user_bias[user_id] += self.learning_rate * error
        self.item_bias[item_id] += self.learning_rate * error
        self.global_bias += self.learning_rate * error * 0.01  # Smaller update for global bias
        
        self.update_count += 1
        
        if self.update_count % 100 == 0:
            logger.debug(f"📊 Embedding update #{self.update_count} (error={error:.4f})")
    
    def get_user_embedding(self, user_id: str) -> NDArray:
        """Get user embedding vector"""
        self._initialize_user(user_id)
        return self.user_embeddings[user_id].copy()
    
    def get_item_embedding(self, item_id: str) -> NDArray:
        """Get item embedding vector"""
        self._initialize_item(item_id)
        return self.item_embeddings[item_id].copy()
    
    def get_similar_items(self, item_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find similar items using cosine similarity
        
        Args:
            item_id: Item identifier
            top_k: Number of similar items to return
        
        Returns:
            List of (item_id, similarity_score) tuples
        """
        if item_id not in self.item_embeddings:
            return []
        
        target_emb = self.item_embeddings[item_id]
        similarities = []
        
        for other_id, other_emb in self.item_embeddings.items():
            if other_id == item_id:
                continue
            
            # Cosine similarity
            if NUMPY_AVAILABLE:
                sim = np.dot(target_emb, other_emb) / \
                      (np.linalg.norm(target_emb) * np.linalg.norm(other_emb))
            else:
                # Fallback: manual cosine similarity
                dot_product = sum(a * b for a, b in zip(target_emb, other_emb))
                norm_target = sum(x ** 2 for x in target_emb) ** 0.5
                norm_other = sum(x ** 2 for x in other_emb) ** 0.5
                sim = dot_product / (norm_target * norm_other) if norm_target and norm_other else 0
            similarities.append((other_id, float(sim)))
        
        # Sort by similarity and return top K
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class OnlineLearningEngine:
    """
    Main online learning engine that coordinates all components
    Processes feedback events and updates models in real-time
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        learning_rate: float = 0.01,
        enable_thompson_sampling: bool = True,
        enable_drift_detection: bool = True
    ):
        """
        Initialize online learning engine
        
        Args:
            embedding_dim: Dimension of embedding vectors
            learning_rate: Learning rate for incremental updates
            enable_thompson_sampling: Whether to use Thompson Sampling
            enable_drift_detection: Whether to enable drift detection
        """
        self.embedding_learner = IncrementalEmbeddingLearner(
            embedding_dim=embedding_dim,
            learning_rate=learning_rate
        )
        
        self.thompson_sampling = None
        if enable_thompson_sampling:
            self.thompson_sampling = ThompsonSampling(n_arms=100)  # Will be dynamic
        
        self.drift_detector = None
        if enable_drift_detection:
            self.drift_detector = ConceptDriftDetector()
        
        # Metrics tracking
        self.total_updates = 0
        self.last_update_time = datetime.now()
        self.user_counts = defaultdict(int)
        self.item_counts = defaultdict(int)
        
        logger.info("✅ OnlineLearningEngine initialized")
    
    async def process_feedback_batch(self, events: List[Dict[str, Any]]):
        """
        Process a batch of feedback events
        
        Args:
            events: List of feedback event dictionaries
        """
        logger.info(f"🔄 Processing batch of {len(events)} feedback events")
        
        for event in events:
            await self.process_feedback_event(event)
        
        logger.info(f"✅ Batch processing complete ({len(events)} events)")
    
    async def process_feedback_event(self, event: Dict[str, Any]):
        """
        Process a single feedback event and update models
        
        Args:
            event: Feedback event dictionary
        """
        user_id = event.get('user_id')
        item_id = event.get('item_id')
        event_type = event.get('event_type')
        metadata = event.get('metadata', {})
        
        # Convert event to reward signal
        reward = self._event_to_reward(event_type, metadata)
        
        # Update embedding learner
        self.embedding_learner.update(user_id, item_id, reward)
        
        # Track prediction error for drift detection
        if self.drift_detector:
            prediction = self.embedding_learner.predict(user_id, item_id)
            error = abs(reward - prediction)
            drift_detected = self.drift_detector.add_observation(error)
            
            if drift_detected:
                logger.warning("🚨 Concept drift detected - consider model retraining")
        
        # Update counts
        self.user_counts[user_id] += 1
        self.item_counts[item_id] += 1
        self.total_updates += 1
        self.last_update_time = datetime.now()
    
    def _event_to_reward(self, event_type: str, metadata: Dict[str, Any]) -> float:
        """
        Convert feedback event to reward signal
        
        Args:
            event_type: Type of event
            metadata: Event metadata
        
        Returns:
            float: Reward value (0-1 range)
        """
        # Define reward mapping
        reward_map = {
            'view': 0.1,
            'click': 0.3,
            'save': 0.6,
            'share': 0.7,
            'conversion': 1.0,
            'rejection': 0.0
        }
        
        base_reward = reward_map.get(event_type, 0.0)
        
        # Adjust based on metadata
        if event_type == 'rating' and 'rating' in metadata:
            # Normalize rating to 0-1 range
            base_reward = metadata['rating'] / 5.0
        
        if 'dwell_time' in metadata:
            # Increase reward for longer dwell times
            dwell_time = metadata['dwell_time']
            if dwell_time > 30:
                base_reward = min(1.0, base_reward * 1.2)
        
        return base_reward
    
    def get_recommendations(
        self,
        user_id: str,
        candidate_items: List[str],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get personalized recommendations for a user
        
        Args:
            user_id: User identifier
            candidate_items: List of candidate item IDs
            top_k: Number of recommendations to return
        
        Returns:
            List of (item_id, score) tuples
        """
        scores = []
        for item_id in candidate_items:
            score = self.embedding_learner.predict(user_id, item_id)
            scores.append((item_id, score))
        
        # Sort by score and return top K
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get engine metrics"""
        return {
            'total_updates': self.total_updates,
            'last_update_time': self.last_update_time.isoformat(),
            'unique_users': len(self.user_counts),
            'unique_items': len(self.item_counts),
            'embedding_dim': self.embedding_learner.embedding_dim,
            'drift_detector_active': self.drift_detector is not None,
            'drift_count': self.drift_detector.drift_count if self.drift_detector else 0
        }


# Global instance
_online_learning_engine = None


def get_online_learning_engine() -> OnlineLearningEngine:
    """Get or create the global online learning engine instance"""
    global _online_learning_engine
    if _online_learning_engine is None:
        _online_learning_engine = OnlineLearningEngine()
    return _online_learning_engine


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        engine = OnlineLearningEngine(embedding_dim=32)
        
        # Simulate feedback events
        events = [
            {'user_id': 'user1', 'item_id': 'place1', 'event_type': 'view', 'metadata': {}},
            {'user_id': 'user1', 'item_id': 'place1', 'event_type': 'click', 'metadata': {}},
            {'user_id': 'user1', 'item_id': 'place1', 'event_type': 'rating', 'metadata': {'rating': 4.5}},
            {'user_id': 'user2', 'item_id': 'place2', 'event_type': 'view', 'metadata': {}},
            {'user_id': 'user2', 'item_id': 'place3', 'event_type': 'save', 'metadata': {}},
        ]
        
        await engine.process_feedback_batch(events)
        
        # Get recommendations
        candidates = ['place1', 'place2', 'place3', 'place4']
        recs = engine.get_recommendations('user1', candidates, top_k=3)
        print("\nRecommendations for user1:")
        for item_id, score in recs:
            print(f"  {item_id}: {score:.4f}")
        
        print("\nEngine metrics:", json.dumps(engine.get_metrics(), indent=2))
    
    asyncio.run(main())
