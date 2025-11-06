"""
Complete LLM + Contextual Bandit Integration
This module shows how to integrate contextual bandits with the existing AI Istanbul LLM system
"""

import logging
from typing import List, Dict, Optional, Any
import numpy as np
from datetime import datetime

# Import existing LLM components
from backend.services.hidden_gems_handler import HiddenGemsHandler
from backend.services.realtime_feedback_loop import get_realtime_feedback_loop

# Import contextual bandit components (Week 11-12)
from backend.ml.bandits.contextual_thompson_sampling import (
    ContextualThompsonSampling,
    BanditContext,
    ContextFeatureExtractor
)
from backend.ml.bandits.persistence import BanditStateManager

# Import online learning (Week 3-4)
from backend.ml.online_learning import OnlineLearningEngine

logger = logging.getLogger(__name__)


class IntegratedRecommendationEngine:
    """
    Complete integration of LLM + Contextual Bandits + Online Learning
    
    Flow:
    1. User query → LLM generates candidates
    2. Extract context features for each candidate
    3. Contextual bandit selects best candidates
    4. Show to user
    5. User feedback updates both basic and contextual bandits
    
    This replaces/enhances the existing recommendation flow
    """
    
    def __init__(
        self,
        redis_url: str,
        enable_contextual_bandits: bool = True,
        enable_basic_bandits: bool = True,
        n_candidates: int = 100
    ):
        """
        Initialize integrated recommendation engine
        
        Args:
            redis_url: Redis connection URL for state persistence
            enable_contextual_bandits: Use contextual bandits (Week 11-12)
            enable_basic_bandits: Use basic Thompson Sampling (Week 3-4)
            n_candidates: Number of candidate arms
        """
        # LLM components (existing)
        self.hidden_gems_handler = HiddenGemsHandler(enable_realtime_learning=True)
        self.feedback_loop = get_realtime_feedback_loop()
        
        # Basic online learning (Week 3-4)
        self.enable_basic_bandits = enable_basic_bandits
        self.online_learning = OnlineLearningEngine() if enable_basic_bandits else None
        
        # Contextual bandits (Week 11-12) ✨ NEW
        self.enable_contextual_bandits = enable_contextual_bandits
        if enable_contextual_bandits:
            self._initialize_contextual_bandits(redis_url, n_candidates)
        else:
            self.contextual_bandit = None
            self.feature_extractor = None
            self.state_manager = None
        
        logger.info(
            f"✅ IntegratedRecommendationEngine initialized: "
            f"contextual_bandits={enable_contextual_bandits}, "
            f"basic_bandits={enable_basic_bandits}"
        )
    
    def _initialize_contextual_bandits(self, redis_url: str, n_candidates: int):
        """Initialize contextual bandit components"""
        # State manager for persistence
        self.state_manager = BanditStateManager(redis_url, key_prefix="ai_istanbul:bandit:")
        
        # Try to load existing bandit
        self.contextual_bandit = self.state_manager.load_bandit('hidden_gems_contextual')
        
        if not self.contextual_bandit:
            # Create new bandit
            self.contextual_bandit = ContextualThompsonSampling(
                n_arms=n_candidates,
                context_dim=20,  # 7 user + 5 item + 4 temporal + 2 interaction (optional)
                alpha=1.0,
                lambda_reg=1.0,
                exploration_bonus=0.1
            )
            logger.info(f"✅ Created new contextual bandit with {n_candidates} arms")
        else:
            logger.info(f"✅ Loaded existing contextual bandit: {self.contextual_bandit.total_pulls} pulls")
        
        # Feature extractor
        self.feature_extractor = ContextFeatureExtractor()
        
        logger.info("✅ Contextual bandit system initialized")
    
    async def get_recommendations(
        self,
        user_query: str,
        user_profile: Dict[str, Any],
        location: Optional[Dict[str, float]] = None,
        top_k: int = 5,
        use_contextual: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get personalized recommendations using LLM + Contextual Bandits
        
        Args:
            user_query: User's search query
            user_profile: User profile data (preferences, history, etc.)
            location: Optional location {'lat': float, 'lng': float}
            top_k: Number of recommendations to return
            use_contextual: Use contextual bandits (True) or basic (False)
        
        Returns:
            List of recommended items with scores and explanations
        """
        # Step 1: LLM generates candidates (existing system)
        llm_candidates = await self._get_llm_candidates(
            user_query=user_query,
            user_profile=user_profile,
            location=location,
            n_candidates=20  # Generate more candidates for selection
        )
        
        if not llm_candidates:
            logger.warning("No LLM candidates generated")
            return []
        
        # Step 2: Score candidates with bandits
        if use_contextual and self.enable_contextual_bandits:
            # Use contextual bandits (Week 11-12) ✨
            scored_candidates = await self._score_with_contextual_bandit(
                candidates=llm_candidates,
                user_profile=user_profile,
                location=location
            )
        elif self.enable_basic_bandits:
            # Use basic Thompson Sampling (Week 3-4)
            scored_candidates = await self._score_with_basic_bandit(
                candidates=llm_candidates,
                user_profile=user_profile
            )
        else:
            # No bandits, use LLM scores only
            scored_candidates = llm_candidates
        
        # Step 3: Sort by score and return top K
        scored_candidates.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        top_recommendations = scored_candidates[:top_k]
        
        # Step 4: Add metadata
        for rec in top_recommendations:
            rec['recommendation_method'] = 'contextual_bandit' if use_contextual else 'basic_bandit'
            rec['timestamp'] = datetime.now().isoformat()
        
        logger.info(
            f"✅ Generated {len(top_recommendations)} recommendations "
            f"(method={'contextual' if use_contextual else 'basic'})"
        )
        
        return top_recommendations
    
    async def _get_llm_candidates(
        self,
        user_query: str,
        user_profile: Dict[str, Any],
        location: Optional[Dict[str, float]],
        n_candidates: int
    ) -> List[Dict[str, Any]]:
        """
        Get candidate recommendations from LLM system
        Uses existing HiddenGemsHandler
        """
        # Use existing LLM system to generate candidates
        # This is the current AI Istanbul recommendation flow
        candidates = []
        
        # Extract context from query
        query_lower = user_query.lower()
        
        # Get all available gems
        all_gems = self.hidden_gems_handler.hidden_gems_db
        
        # Flatten all gems
        for neighborhood, gems in all_gems.items():
            for gem in gems:
                # Add neighborhood and basic metadata
                candidate = {
                    'id': f"{neighborhood}_{gem['name'].lower().replace(' ', '_')}",
                    'name': gem['name'],
                    'neighborhood': neighborhood,
                    'type': gem.get('type', 'general'),
                    'description': gem.get('description', ''),
                    'category': gem.get('type', 'general'),
                    'llm_score': 0.5,  # Base score
                    'rating': gem.get('rating', 4.0),
                    'price': gem.get('cost', '₺₺'),
                    'review_count': gem.get('popularity', 100),
                    'distance': self._calculate_distance(gem, location) if location else 5.0
                }
                candidates.append(candidate)
        
        # Simple LLM-style scoring based on query match
        for candidate in candidates:
            score = 0.5  # Base score
            
            # Keyword matching (simple LLM simulation)
            if any(word in candidate['description'].lower() for word in query_lower.split()):
                score += 0.2
            if any(word in candidate['name'].lower() for word in query_lower.split()):
                score += 0.3
            
            # User preference matching
            if user_profile.get('preferred_types'):
                if candidate['type'] in user_profile['preferred_types']:
                    score += 0.2
            
            candidate['llm_score'] = min(score, 1.0)
        
        # Sort by LLM score and return top N
        candidates.sort(key=lambda x: x['llm_score'], reverse=True)
        return candidates[:n_candidates]
    
    async def _score_with_contextual_bandit(
        self,
        candidates: List[Dict[str, Any]],
        user_profile: Dict[str, Any],
        location: Optional[Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """
        Score candidates using contextual bandits (Week 11-12)
        """
        scored_candidates = []
        
        # Get user interaction history if available
        interaction_history = user_profile.get('interaction_history', [])
        
        for idx, candidate in enumerate(candidates):
            # Extract context for this candidate
            context = self.feature_extractor.extract_context(
                user_profile=user_profile,
                item=candidate,
                interaction_history=interaction_history if interaction_history else None
            )
            
            # Map candidate to arm (use index)
            arm_idx = idx % self.contextual_bandit.n_arms
            
            # Get contextual bandit score
            # Note: select_arm returns the best arm, but we want the score
            # So we compute the expected reward directly
            context_vec = context.to_vector()
            if len(context_vec) != self.contextual_bandit.context_dim:
                # Pad or truncate
                if len(context_vec) < self.contextual_bandit.context_dim:
                    context_vec = np.pad(
                        context_vec,
                        (0, self.contextual_bandit.context_dim - len(context_vec))
                    )
                else:
                    context_vec = context_vec[:self.contextual_bandit.context_dim]
            
            # Get expected reward from posterior mean
            theta_mean, _ = self.contextual_bandit._get_theta(arm_idx)
            contextual_score = float(context_vec @ theta_mean)
            
            # Normalize to 0-1
            contextual_score = 1.0 / (1.0 + np.exp(-contextual_score))  # Sigmoid
            
            # Combine LLM score with contextual score
            llm_score = candidate.get('llm_score', 0.5)
            final_score = 0.5 * llm_score + 0.5 * contextual_score
            
            # Add to candidate
            candidate['contextual_score'] = contextual_score
            candidate['final_score'] = final_score
            candidate['arm_idx'] = arm_idx
            candidate['context'] = context  # Store for feedback update
            
            scored_candidates.append(candidate)
        
        return scored_candidates
    
    async def _score_with_basic_bandit(
        self,
        candidates: List[Dict[str, Any]],
        user_profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Score candidates using basic Thompson Sampling (Week 3-4)
        """
        scored_candidates = []
        
        for idx, candidate in enumerate(candidates):
            # Use online learning to get score
            user_id = user_profile.get('user_id', 'anonymous')
            item_id = candidate.get('id', str(idx))
            
            # Get incremental MF score if available
            imf_score = 0.5
            if self.online_learning and hasattr(self.online_learning, 'embedding_learner'):
                try:
                    imf_score = self.online_learning.embedding_learner.predict(user_id, item_id)
                except:
                    pass
            
            # Get Thompson Sampling exploration bonus
            thompson_bonus = 0.0
            if self.online_learning and hasattr(self.online_learning, 'thompson_sampling'):
                arm = idx % 100  # Map to arm
                # Sample from Beta distribution for exploration
                try:
                    thompson_bonus = np.random.beta(
                        self.online_learning.thompson_sampling.alpha[arm],
                        self.online_learning.thompson_sampling.beta[arm]
                    ) * 0.1  # Small exploration bonus
                except:
                    pass
            
            # Combine scores
            llm_score = candidate.get('llm_score', 0.5)
            final_score = 0.5 * llm_score + 0.3 * imf_score + 0.2 * thompson_bonus
            
            candidate['imf_score'] = imf_score
            candidate['thompson_bonus'] = thompson_bonus
            candidate['final_score'] = final_score
            candidate['arm_idx'] = idx % 100
            
            scored_candidates.append(candidate)
        
        return scored_candidates
    
    async def process_feedback(
        self,
        user_id: str,
        item_id: str,
        feedback_type: str,
        recommendation: Dict[str, Any],
        user_profile: Dict[str, Any]
    ):
        """
        Process user feedback and update both bandit systems
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            feedback_type: Type of feedback ('view', 'click', 'like', 'booking')
            recommendation: The original recommendation dict (contains context)
            user_profile: User profile data
        """
        # Convert feedback to reward
        reward_map = {
            'view': 0.2,
            'click': 0.5,
            'like': 0.8,
            'booking': 1.0,
            'skip': 0.0
        }
        reward = reward_map.get(feedback_type, 0.0)
        
        # Update contextual bandit (Week 11-12)
        if self.enable_contextual_bandits and 'context' in recommendation:
            arm_idx = recommendation.get('arm_idx', 0)
            context = recommendation['context']
            
            self.contextual_bandit.update(arm_idx, context, reward)
            
            # Save state periodically (every 10 updates)
            if self.contextual_bandit.total_pulls % 10 == 0:
                self.state_manager.save_bandit(self.contextual_bandit, 'hidden_gems_contextual')
            
            logger.debug(
                f"✅ Updated contextual bandit: user={user_id}, item={item_id}, "
                f"reward={reward}, arm={arm_idx}"
            )
        
        # Update basic bandit (Week 3-4)
        if self.enable_basic_bandits and self.online_learning:
            arm_idx = recommendation.get('arm_idx', 0)
            
            # Update Thompson Sampling
            if hasattr(self.online_learning, 'thompson_sampling'):
                self.online_learning.thompson_sampling.update(arm_idx, reward)
            
            # Update Incremental MF
            if hasattr(self.online_learning, 'embedding_learner'):
                try:
                    self.online_learning.embedding_learner.update(user_id, item_id, reward)
                except Exception as e:
                    logger.warning(f"Failed to update embedding learner: {e}")
            
            logger.debug(
                f"✅ Updated basic bandit: user={user_id}, item={item_id}, reward={reward}"
            )
        
        # Also update existing feedback loop
        if self.feedback_loop:
            try:
                await self.feedback_loop.process_event({
                    'user_id': user_id,
                    'item_id': item_id,
                    'event_type': feedback_type,
                    'reward': reward,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.warning(f"Failed to update feedback loop: {e}")
    
    def _calculate_distance(
        self,
        item: Dict[str, Any],
        location: Optional[Dict[str, float]]
    ) -> float:
        """Calculate distance between user location and item"""
        if not location:
            return 5.0  # Default distance
        
        # Simplified distance calculation
        # In production, use proper geo-distance
        item_lat = item.get('latitude', 41.0082)  # Istanbul center default
        item_lng = item.get('longitude', 28.9784)
        
        user_lat = location.get('lat', 41.0082)
        user_lng = location.get('lng', 28.9784)
        
        # Simple Euclidean distance (not accurate for lat/lng but good enough)
        distance = np.sqrt((item_lat - user_lat)**2 + (item_lng - user_lng)**2) * 111  # Rough km conversion
        
        return float(distance)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from all bandit systems"""
        stats = {
            'timestamp': datetime.now().isoformat(),
            'contextual_bandits_enabled': self.enable_contextual_bandits,
            'basic_bandits_enabled': self.enable_basic_bandits
        }
        
        # Contextual bandit stats
        if self.enable_contextual_bandits and self.contextual_bandit:
            contextual_stats = self.contextual_bandit.get_arm_stats()
            stats['contextual_bandit'] = {
                'total_pulls': int(self.contextual_bandit.total_pulls),
                'n_arms': self.contextual_bandit.n_arms,
                'avg_reward': float(np.mean([s.get('avg_reward', 0) for s in contextual_stats if s.get('pulls', 0) > 0] or [0])),
                'exploration_rate': float(np.sum([s.get('pulls', 0) < 10 for s in contextual_stats]) / len(contextual_stats) if contextual_stats else 0)
            }
        
        # Basic bandit stats
        if self.enable_basic_bandits and self.online_learning:
            if hasattr(self.online_learning, 'thompson_sampling'):
                try:
                    ts_stats = self.online_learning.thompson_sampling.get_arm_stats()
                    stats['basic_thompson_sampling'] = {
                        'total_pulls': sum(s.get('pulls', 0) for s in ts_stats),
                        'n_arms': len(ts_stats),
                        'avg_reward': float(np.mean([s.get('avg_reward', 0) for s in ts_stats if s.get('pulls', 0) > 0] or [0]))
                    }
                except Exception as e:
                    logger.warning(f"Failed to get basic Thompson stats: {e}")
        
        return stats


# Example usage and integration
async def example_usage():
    """Example of how to use the integrated system"""
    import os
    
    # Initialize
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
    engine = IntegratedRecommendationEngine(
        redis_url=redis_url,
        enable_contextual_bandits=True,  # Week 11-12 ✨
        enable_basic_bandits=True,        # Week 3-4
        n_candidates=100
    )
    
    # User profile
    user_profile = {
        'user_id': 'user_123',
        'preferred_cuisines': ['Turkish', 'Mediterranean'],
        'preferred_types': ['nature', 'historical'],
        'budget': 75.0,
        'location': {'lat': 41.0082, 'lng': 28.9784},
        'interaction_count': 15,
        'interaction_history': []
    }
    
    # Get recommendations
    recommendations = await engine.get_recommendations(
        user_query="hidden beach near Sarıyer",
        user_profile=user_profile,
        location={'lat': 41.1, 'lng': 29.0},
        top_k=5,
        use_contextual=True  # Use contextual bandits
    )
    
    # Show recommendations
    print("\n🎯 Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['name']}")
        print(f"   Score: {rec['final_score']:.3f} (LLM: {rec.get('llm_score', 0):.3f}, Contextual: {rec.get('contextual_score', 0):.3f})")
        print(f"   Type: {rec['type']}, Neighborhood: {rec['neighborhood']}")
        print()
    
    # Simulate user feedback
    if recommendations:
        selected_rec = recommendations[0]
        await engine.process_feedback(
            user_id='user_123',
            item_id=selected_rec['id'],
            feedback_type='click',
            recommendation=selected_rec,
            user_profile=user_profile
        )
        print("✅ Feedback processed")
    
    # Get stats
    stats = engine.get_stats()
    print("\n📊 System Stats:")
    print(f"Contextual Bandit: {stats.get('contextual_bandit', {})}")
    print(f"Basic Thompson: {stats.get('basic_thompson_sampling', {})}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
