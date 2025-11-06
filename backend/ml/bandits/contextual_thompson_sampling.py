"""
Contextual Thompson Sampling for Personalized Recommendations
Extends basic Thompson Sampling with context-aware selection

This implementation is BUDGET-OPTIMIZED:
- Linear models (no neural networks)
- NumPy-based computations (no GPU needed)
- Fast inference (<5ms per selection)
- Efficient memory usage
- Seamless integration with existing LLM system
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class BanditContext:
    """
    Context features for contextual bandit selection
    
    Attributes:
        user_features: User preference features (cuisine, budget, location, etc.)
        item_features: Item characteristic features (category, price, rating, etc.)
        temporal_features: Time-based features (hour, day_of_week, season)
        interaction_features: Previous interaction patterns
    """
    user_features: np.ndarray
    item_features: np.ndarray
    temporal_features: np.ndarray
    interaction_features: Optional[np.ndarray] = None
    
    def to_vector(self) -> np.ndarray:
        """Combine all features into a single context vector"""
        features = [self.user_features, self.item_features, self.temporal_features]
        if self.interaction_features is not None:
            features.append(self.interaction_features)
        return np.concatenate(features)
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'BanditContext':
        """Create context from dictionary"""
        return BanditContext(
            user_features=np.array(data.get('user_features', [])),
            item_features=np.array(data.get('item_features', [])),
            temporal_features=np.array(data.get('temporal_features', [])),
            interaction_features=np.array(data.get('interaction_features', [])) 
                if data.get('interaction_features') else None
        )


class ContextualThompsonSampling:
    """
    Contextual Thompson Sampling with Linear Models
    
    Features:
    - Context-aware arm selection using linear regression
    - Bayesian posterior updating
    - Efficient matrix operations
    - Cold-start handling for new items
    - Exploration bonus tuning
    
    Integration with LLM System:
    1. LLM generates candidate recommendations
    2. Contextual bandit selects which to show based on context
    3. User feedback updates bandit parameters
    4. System learns optimal exploration-exploitation balance
    """
    
    def __init__(
        self, 
        n_arms: int, 
        context_dim: int = 20,
        alpha: float = 1.0,
        lambda_reg: float = 1.0,
        exploration_bonus: float = 0.1
    ):
        """
        Initialize Contextual Thompson Sampling
        
        Args:
            n_arms: Number of arms (recommendation candidates)
            context_dim: Dimensionality of context features
            alpha: Exploration parameter (higher = more exploration)
            lambda_reg: L2 regularization parameter
            exploration_bonus: Additional exploration bonus for under-explored arms
        """
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.exploration_bonus = exploration_bonus
        
        # Linear model parameters per arm (θ ~ N(μ, Σ))
        # A = X^T X + λI (precision matrix)
        # b = X^T y (weighted rewards)
        self.A = {arm: np.eye(context_dim) * lambda_reg for arm in range(n_arms)}
        self.b = {arm: np.zeros(context_dim) for arm in range(n_arms)}
        
        # Track statistics
        self.arm_pulls = np.zeros(n_arms)
        self.arm_rewards = np.zeros(n_arms)
        self.total_pulls = 0
        
        # Cache for efficiency
        self._theta_cache = {}
        self._cache_valid = {arm: False for arm in range(n_arms)}
        
        logger.info(
            f"✅ ContextualThompsonSampling initialized: "
            f"n_arms={n_arms}, context_dim={context_dim}, "
            f"alpha={alpha}, exploration_bonus={exploration_bonus}"
        )
    
    def _get_theta(self, arm: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get posterior mean and covariance for arm's theta
        
        Returns:
            (theta_mean, theta_cov): Posterior distribution parameters
        """
        if self._cache_valid[arm] and arm in self._theta_cache:
            return self._theta_cache[arm]
        
        # Compute posterior: θ ~ N(A^(-1)b, A^(-1))
        try:
            A_inv = np.linalg.inv(self.A[arm])
            theta_mean = A_inv @ self.b[arm]
            theta_cov = self.alpha * A_inv
            
            self._theta_cache[arm] = (theta_mean, theta_cov)
            self._cache_valid[arm] = True
            
            return theta_mean, theta_cov
        except np.linalg.LinAlgError:
            # Fallback for singular matrix
            logger.warning(f"Singular matrix for arm {arm}, using regularization")
            theta_mean = np.zeros(self.context_dim)
            theta_cov = np.eye(self.context_dim) * self.alpha
            return theta_mean, theta_cov
    
    def select_arm(self, context: BanditContext, candidate_arms: Optional[List[int]] = None) -> int:
        """
        Select best arm given context using Thompson Sampling
        
        Args:
            context: Current context features
            candidate_arms: Subset of arms to consider (None = all arms)
        
        Returns:
            Selected arm index
        """
        context_vec = context.to_vector()
        
        # Ensure context dimension matches
        if len(context_vec) != self.context_dim:
            logger.warning(
                f"Context dimension mismatch: expected {self.context_dim}, "
                f"got {len(context_vec)}. Padding/truncating."
            )
            if len(context_vec) < self.context_dim:
                context_vec = np.pad(context_vec, (0, self.context_dim - len(context_vec)))
            else:
                context_vec = context_vec[:self.context_dim]
        
        arms_to_consider = candidate_arms if candidate_arms else list(range(self.n_arms))
        
        # Sample θ from posterior and compute expected reward for each arm
        arm_scores = {}
        for arm in arms_to_consider:
            theta_mean, theta_cov = self._get_theta(arm)
            
            # Sample from posterior distribution
            try:
                sampled_theta = np.random.multivariate_normal(theta_mean, theta_cov)
            except:
                # Fallback for numerical issues
                sampled_theta = theta_mean + np.random.randn(self.context_dim) * np.sqrt(self.alpha)
            
            # Compute expected reward
            expected_reward = context_vec @ sampled_theta
            
            # Add exploration bonus for under-explored arms
            if self.total_pulls > 0:
                exploration = self.exploration_bonus * np.sqrt(
                    np.log(self.total_pulls + 1) / (self.arm_pulls[arm] + 1)
                )
                expected_reward += exploration
            
            arm_scores[arm] = expected_reward
        
        # Select arm with highest score
        selected_arm = max(arm_scores, key=arm_scores.get)
        
        logger.debug(
            f"Arm selection: arm={selected_arm}, "
            f"score={arm_scores[selected_arm]:.4f}, "
            f"pulls={int(self.arm_pulls[selected_arm])}"
        )
        
        return selected_arm
    
    def update(self, arm: int, context: BanditContext, reward: float):
        """
        Update arm parameters based on observed reward
        
        Args:
            arm: Arm that was selected
            context: Context in which arm was selected
            reward: Observed reward (0-1 range recommended)
        """
        context_vec = context.to_vector()
        
        # Ensure dimension match
        if len(context_vec) != self.context_dim:
            if len(context_vec) < self.context_dim:
                context_vec = np.pad(context_vec, (0, self.context_dim - len(context_vec)))
            else:
                context_vec = context_vec[:self.context_dim]
        
        # Update precision matrix and weighted rewards
        # A ← A + x x^T
        # b ← b + r x
        self.A[arm] += np.outer(context_vec, context_vec)
        self.b[arm] += reward * context_vec
        
        # Update statistics
        self.arm_pulls[arm] += 1
        self.arm_rewards[arm] += reward
        self.total_pulls += 1
        
        # Invalidate cache for this arm
        self._cache_valid[arm] = False
        
        if self.total_pulls % 100 == 0:
            avg_reward = np.sum(self.arm_rewards) / self.total_pulls
            logger.info(
                f"📊 Contextual bandit update #{self.total_pulls}: "
                f"avg_reward={avg_reward:.4f}, "
                f"arm={arm}, reward={reward:.4f}"
            )
    
    def get_arm_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for all arms"""
        stats = []
        for arm in range(self.n_arms):
            theta_mean, theta_cov = self._get_theta(arm)
            
            avg_reward = (
                self.arm_rewards[arm] / self.arm_pulls[arm]
                if self.arm_pulls[arm] > 0
                else 0.0
            )
            
            # Compute confidence (based on uncertainty)
            uncertainty = np.trace(theta_cov) / self.context_dim
            
            stats.append({
                'arm': arm,
                'pulls': int(self.arm_pulls[arm]),
                'total_reward': float(self.arm_rewards[arm]),
                'avg_reward': float(avg_reward),
                'uncertainty': float(uncertainty),
                'theta_norm': float(np.linalg.norm(theta_mean))
            })
        
        return stats
    
    def get_best_arm(self, context: BanditContext) -> int:
        """
        Get best arm without exploration (pure exploitation)
        Useful for evaluation or final recommendations
        
        Args:
            context: Current context features
        
        Returns:
            Best arm index based on expected reward
        """
        context_vec = context.to_vector()
        
        if len(context_vec) != self.context_dim:
            if len(context_vec) < self.context_dim:
                context_vec = np.pad(context_vec, (0, self.context_dim - len(context_vec)))
            else:
                context_vec = context_vec[:self.context_dim]
        
        arm_values = {}
        for arm in range(self.n_arms):
            theta_mean, _ = self._get_theta(arm)
            expected_reward = context_vec @ theta_mean
            arm_values[arm] = expected_reward
        
        return max(arm_values, key=arm_values.get)
    
    def save_state(self) -> Dict[str, Any]:
        """Save bandit state for persistence"""
        return {
            'n_arms': self.n_arms,
            'context_dim': self.context_dim,
            'alpha': self.alpha,
            'lambda_reg': self.lambda_reg,
            'exploration_bonus': self.exploration_bonus,
            'A': {arm: self.A[arm].tolist() for arm in range(self.n_arms)},
            'b': {arm: self.b[arm].tolist() for arm in range(self.n_arms)},
            'arm_pulls': self.arm_pulls.tolist(),
            'arm_rewards': self.arm_rewards.tolist(),
            'total_pulls': int(self.total_pulls),
            'timestamp': datetime.now().isoformat()
        }
    
    @classmethod
    def load_state(cls, state: Dict[str, Any]) -> 'ContextualThompsonSampling':
        """Load bandit from saved state"""
        bandit = cls(
            n_arms=state['n_arms'],
            context_dim=state['context_dim'],
            alpha=state['alpha'],
            lambda_reg=state['lambda_reg'],
            exploration_bonus=state['exploration_bonus']
        )
        
        # Restore parameters
        bandit.A = {int(arm): np.array(A) for arm, A in state['A'].items()}
        bandit.b = {int(arm): np.array(b) for arm, b in state['b'].items()}
        bandit.arm_pulls = np.array(state['arm_pulls'])
        bandit.arm_rewards = np.array(state['arm_rewards'])
        bandit.total_pulls = state['total_pulls']
        
        logger.info(f"✅ Loaded ContextualThompsonSampling with {bandit.total_pulls} pulls")
        
        return bandit


class ContextFeatureExtractor:
    """
    Extract context features for bandit from user profile and item data
    Integrates with existing LLM system
    """
    
    def __init__(self):
        """Initialize feature extractor"""
        self.cuisine_map = {}
        self.category_map = {}
        self.next_cuisine_id = 0
        self.next_category_id = 0
        
        logger.info("✅ ContextFeatureExtractor initialized")
    
    def extract_user_features(self, user_profile: Dict[str, Any]) -> np.ndarray:
        """
        Extract user features from profile
        
        Features:
        - Cuisine preferences (one-hot encoded)
        - Budget level (normalized)
        - Location features (lat, lng)
        - Interaction history summary
        """
        features = []
        
        # Cuisine preferences (top 3)
        cuisines = user_profile.get('preferred_cuisines', [])[:3]
        cuisine_features = np.zeros(3)
        for i, cuisine in enumerate(cuisines):
            if cuisine not in self.cuisine_map:
                self.cuisine_map[cuisine] = self.next_cuisine_id
                self.next_cuisine_id += 1
            cuisine_features[i] = self.cuisine_map[cuisine]
        features.extend(cuisine_features)
        
        # Budget (normalized to 0-1)
        budget = user_profile.get('budget', 50.0) / 200.0  # Assume max 200
        features.append(budget)
        
        # Location (normalized)
        location = user_profile.get('location', {'lat': 41.0082, 'lng': 28.9784})  # Istanbul center
        features.extend([
            location['lat'] / 90.0,  # Normalize latitude
            location['lng'] / 180.0  # Normalize longitude
        ])
        
        # Interaction count (log-scaled)
        interaction_count = user_profile.get('interaction_count', 0)
        features.append(np.log1p(interaction_count))
        
        return np.array(features)
    
    def extract_item_features(self, item: Dict[str, Any]) -> np.ndarray:
        """
        Extract item features
        
        Features:
        - Category (encoded)
        - Price level (normalized)
        - Rating (normalized)
        - Popularity (log-scaled)
        - Distance (if available)
        """
        features = []
        
        # Category
        category = item.get('category', 'restaurant')
        if category not in self.category_map:
            self.category_map[category] = self.next_category_id
            self.next_category_id += 1
        features.append(self.category_map[category])
        
        # Price (normalized to 0-1)
        # Handle string prices like "₺₺" or numeric prices
        price_raw = item.get('price', 50.0)
        if isinstance(price_raw, str):
            # Convert ₺ symbols to numeric (₺ = 25, ₺₺ = 50, ₺₺₺ = 75, ₺₺₺₺ = 100)
            price = len(price_raw) * 25.0 / 200.0
        else:
            price = float(price_raw) / 200.0
        features.append(price)
        
        # Rating (already 0-5, normalize to 0-1)
        rating = item.get('rating', 3.0) / 5.0
        features.append(rating)
        
        # Review count (log-scaled)
        review_count = item.get('review_count', 0)
        features.append(np.log1p(review_count))
        
        # Distance (if available, normalized to km)
        distance = item.get('distance', 5.0) / 20.0  # Assume max 20km
        features.append(distance)
        
        return np.array(features)
    
    def extract_temporal_features(self) -> np.ndarray:
        """
        Extract temporal features
        
        Features:
        - Hour of day (0-23, normalized)
        - Day of week (0-6, normalized)
        - Is weekend (binary)
        - Season (encoded)
        """
        now = datetime.now()
        
        features = [
            now.hour / 23.0,  # Hour
            now.weekday() / 6.0,  # Day of week
            float(now.weekday() >= 5),  # Is weekend
            (now.month % 12) / 11.0  # Season (month-based)
        ]
        
        return np.array(features)
    
    def extract_context(
        self, 
        user_profile: Dict[str, Any], 
        item: Dict[str, Any],
        interaction_history: Optional[List[Dict]] = None
    ) -> BanditContext:
        """
        Extract complete context for bandit
        
        Args:
            user_profile: User profile data
            item: Item/restaurant data
            interaction_history: Recent user interactions
        
        Returns:
            BanditContext with all features
        """
        user_features = self.extract_user_features(user_profile)
        item_features = self.extract_item_features(item)
        temporal_features = self.extract_temporal_features()
        
        # Extract interaction features if available
        interaction_features = None
        if interaction_history:
            # Simple aggregation: avg rating, interaction count
            avg_rating = np.mean([i.get('rating', 0) for i in interaction_history])
            count = len(interaction_history)
            interaction_features = np.array([avg_rating / 5.0, np.log1p(count)])
        
        return BanditContext(
            user_features=user_features,
            item_features=item_features,
            temporal_features=temporal_features,
            interaction_features=interaction_features
        )


# Example usage and integration with LLM system
if __name__ == "__main__":
    # Example: Initialize contextual bandit
    n_candidates = 10  # Number of LLM-generated candidates
    context_dim = 20   # Total feature dimension
    
    bandit = ContextualThompsonSampling(
        n_arms=n_candidates,
        context_dim=context_dim,
        exploration_bonus=0.1
    )
    
    feature_extractor = ContextFeatureExtractor()
    
    # Simulate recommendation flow
    user_profile = {
        'preferred_cuisines': ['Turkish', 'Mediterranean', 'Italian'],
        'budget': 75.0,
        'location': {'lat': 41.0082, 'lng': 28.9784},
        'interaction_count': 15
    }
    
    item = {
        'category': 'restaurant',
        'price': 80.0,
        'rating': 4.5,
        'review_count': 245,
        'distance': 2.3
    }
    
    # Extract context
    context = feature_extractor.extract_context(user_profile, item)
    
    # Select arm
    selected_arm = bandit.select_arm(context)
    print(f"Selected arm: {selected_arm}")
    
    # Simulate feedback and update
    reward = 1.0  # User clicked/liked the recommendation
    bandit.update(selected_arm, context, reward)
    
    # Get statistics
    stats = bandit.get_arm_stats()
    print(f"\nArm statistics:")
    for stat in stats[:3]:  # Show first 3
        print(f"  Arm {stat['arm']}: avg_reward={stat['avg_reward']:.3f}, pulls={stat['pulls']}")
