"""
Integrated Recommendation Service
Combines LLM, NCF, and LightGBM ranker for production recommendations

Architecture:
1. LLM generates initial candidates based on user query/context
2. NCF provides collaborative filtering scores
3. LightGBM re-ranks with rich features
4. Final ensemble combines all signals
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import time
from datetime import datetime
import logging

# Add ML directory to path
ml_dir = Path(__file__).parent.parent
sys.path.insert(0, str(ml_dir))

from models.lightgbm_ranker import LightGBMRanker
from serving.lightgbm_ranker_inference import RankerInferenceService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntegratedRecommendationService:
    """
    Integrated service combining LLM, NCF, and LightGBM
    """
    
    def __init__(
        self,
        lightgbm_model_path: Optional[str] = None,
        lightgbm_feature_stats_path: Optional[str] = None,
        ncf_model_path: Optional[str] = None,
        ncf_embeddings_path: Optional[str] = None,
        ensemble_weights: Optional[Dict[str, float]] = None,
        use_llm: bool = True,
        cache_size: int = 1000
    ):
        """
        Initialize integrated recommendation service
        
        Args:
            lightgbm_model_path: Path to LightGBM model
            lightgbm_feature_stats_path: Path to feature statistics
            ncf_model_path: Path to NCF model
            ncf_embeddings_path: Path to NCF embeddings
            ensemble_weights: Weights for each component
            use_llm: Whether to use LLM for candidate generation
            cache_size: Cache size for statistics
        """
        logger.info("🚀 Initializing Integrated Recommendation Service...")
        
        # Component flags
        self.has_lightgbm = False
        self.has_ncf = False
        self.use_llm = use_llm
        
        # Initialize LightGBM ranker
        if lightgbm_model_path and lightgbm_feature_stats_path:
            try:
                self.lightgbm_service = RankerInferenceService(
                    model_path=lightgbm_model_path,
                    feature_stats_path=lightgbm_feature_stats_path,
                    ncf_embeddings_path=ncf_embeddings_path,
                    cache_size=cache_size
                )
                self.has_lightgbm = True
                logger.info("✅ LightGBM ranker loaded")
            except Exception as e:
                logger.warning(f"⚠️  LightGBM ranker not available: {e}")
        
        # Initialize NCF (if available)
        if ncf_model_path:
            try:
                # Import NCF inference service
                from serving.lightweight_ncf_inference import NCFInferenceService
                self.ncf_service = NCFInferenceService(ncf_model_path)
                self.has_ncf = True
                logger.info("✅ NCF model loaded")
            except Exception as e:
                logger.warning(f"⚠️  NCF model not available: {e}")
        
        # Ensemble weights (defaults)
        self.ensemble_weights = ensemble_weights or {
            'llm': 0.4,      # LLM semantic matching
            'ncf': 0.3,      # Collaborative filtering
            'lightgbm': 0.3  # Feature-based ranking
        }
        
        # Normalize weights
        total_weight = sum(self.ensemble_weights.values())
        self.ensemble_weights = {
            k: v / total_weight for k, v in self.ensemble_weights.items()
        }
        
        logger.info(f"📊 Ensemble weights: {self.ensemble_weights}")
        logger.info("✅ Integrated recommendation service ready!")
    
    async def get_recommendations(
        self,
        user_id: int,
        query: str,
        context: Dict[str, Any],
        top_k: int = 10,
        candidate_pool_size: int = 100,
        user_stats: Optional[Dict] = None,
        item_stats: Optional[Dict[int, Dict]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get integrated recommendations
        
        Args:
            user_id: User ID
            query: User query/request
            context: Context information (location, time, preferences, etc.)
            top_k: Number of recommendations to return
            candidate_pool_size: Size of candidate pool for re-ranking
            user_stats: Optional user statistics
            item_stats: Optional item statistics
            
        Returns:
            List of recommended items with scores and explanations
        """
        start_time = time.time()
        
        logger.info(f"🎯 Getting recommendations for user {user_id}: '{query}'")
        
        # Step 1: Generate candidates using LLM
        candidates = await self._get_llm_candidates(
            user_id=user_id,
            query=query,
            context=context,
            top_k=candidate_pool_size
        )
        
        if not candidates:
            logger.warning("⚠️  No candidates from LLM, returning empty list")
            return []
        
        candidate_ids = [c['item_id'] for c in candidates]
        logger.info(f"📦 Generated {len(candidate_ids)} candidates from LLM")
        
        # Step 2: Score candidates with all available models
        scores = {}
        
        # LLM scores (from initial retrieval)
        scores['llm'] = {c['item_id']: c['score'] for c in candidates}
        
        # NCF scores
        if self.has_ncf:
            ncf_scores = self._get_ncf_scores(user_id, candidate_ids)
            scores['ncf'] = ncf_scores
            logger.info(f"✅ NCF scores computed")
        
        # LightGBM scores
        if self.has_lightgbm:
            lightgbm_scores = self._get_lightgbm_scores(
                user_id=user_id,
                candidate_ids=candidate_ids,
                user_stats=user_stats,
                item_stats=item_stats
            )
            scores['lightgbm'] = lightgbm_scores
            logger.info(f"✅ LightGBM scores computed")
        
        # Step 3: Ensemble scores
        final_scores = self._ensemble_scores(scores, candidate_ids)
        
        # Step 4: Re-rank and select top K
        ranked_items = sorted(
            final_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # Step 5: Enrich with details and explanations
        recommendations = await self._enrich_recommendations(
            ranked_items=ranked_items,
            candidates=candidates,
            scores=scores,
            query=query,
            context=context
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Generated {len(recommendations)} recommendations in {elapsed_time:.3f}s")
        
        return recommendations
    
    async def _get_llm_candidates(
        self,
        user_id: int,
        query: str,
        context: Dict[str, Any],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Get candidate items from LLM system
        
        This integrates with the existing AI Istanbul LLM handler
        """
        if not self.use_llm:
            # Fallback: return popular items or random items
            return self._get_fallback_candidates(top_k)
        
        try:
            # Import the existing AI Istanbul handler
            from advanced_istanbul_ai import AdvancedIstanbulAI
            
            # Initialize AI handler (or reuse existing instance)
            ai_handler = AdvancedIstanbulAI()
            
            # Process query with LLM
            llm_response = await ai_handler.process_query(
                query=query,
                user_context=context,
                user_id=str(user_id)
            )
            
            # Extract candidates from LLM response
            candidates = self._extract_candidates_from_llm_response(
                llm_response,
                top_k
            )
            
            return candidates
            
        except Exception as e:
            logger.error(f"❌ Error getting LLM candidates: {e}")
            return self._get_fallback_candidates(top_k)
    
    def _extract_candidates_from_llm_response(
        self,
        llm_response: Dict[str, Any],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Extract candidate items from LLM response
        """
        candidates = []
        
        # Extract from recommendations field
        if 'recommendations' in llm_response:
            for idx, rec in enumerate(llm_response['recommendations'][:top_k]):
                candidates.append({
                    'item_id': rec.get('id', idx),
                    'name': rec.get('name', ''),
                    'category': rec.get('category', ''),
                    'score': rec.get('confidence', 1.0) or 1.0 - (idx * 0.05),
                    'llm_explanation': rec.get('explanation', ''),
                    'metadata': rec
                })
        
        # Extract from hidden gems
        elif 'hidden_gems' in llm_response:
            for idx, gem in enumerate(llm_response['hidden_gems'][:top_k]):
                candidates.append({
                    'item_id': gem.get('id', idx),
                    'name': gem.get('name', ''),
                    'category': gem.get('category', 'attraction'),
                    'score': gem.get('score', 1.0) or 1.0 - (idx * 0.05),
                    'llm_explanation': gem.get('reason', ''),
                    'metadata': gem
                })
        
        return candidates
    
    def _get_fallback_candidates(self, top_k: int) -> List[Dict[str, Any]]:
        """
        Get fallback candidates when LLM is not available
        """
        # Return mock candidates (in production, query database for popular items)
        return [
            {
                'item_id': i,
                'name': f'Item {i}',
                'category': 'attraction',
                'score': 1.0 - (i * 0.05),
                'llm_explanation': 'Popular item',
                'metadata': {}
            }
            for i in range(top_k)
        ]
    
    def _get_ncf_scores(
        self,
        user_id: int,
        item_ids: List[int]
    ) -> Dict[int, float]:
        """
        Get NCF collaborative filtering scores
        """
        if not self.has_ncf:
            return {}
        
        try:
            # Get NCF predictions
            scores = self.ncf_service.predict_batch(
                user_ids=[user_id] * len(item_ids),
                item_ids=item_ids
            )
            
            return dict(zip(item_ids, scores))
            
        except Exception as e:
            logger.error(f"❌ Error getting NCF scores: {e}")
            return {}
    
    def _get_lightgbm_scores(
        self,
        user_id: int,
        candidate_ids: List[int],
        user_stats: Optional[Dict],
        item_stats: Optional[Dict[int, Dict]]
    ) -> Dict[int, float]:
        """
        Get LightGBM ranking scores
        """
        if not self.has_lightgbm:
            return {}
        
        try:
            # Get LightGBM rankings
            ranked_items = self.lightgbm_service.rank_items_for_user(
                user_id=user_id,
                item_ids=candidate_ids,
                user_stats=user_stats,
                item_stats=item_stats
            )
            
            return {item_id: score for item_id, score in ranked_items}
            
        except Exception as e:
            logger.error(f"❌ Error getting LightGBM scores: {e}")
            return {}
    
    def _ensemble_scores(
        self,
        scores: Dict[str, Dict[int, float]],
        candidate_ids: List[int]
    ) -> Dict[int, float]:
        """
        Combine scores from different models using ensemble weights
        """
        final_scores = {}
        
        for item_id in candidate_ids:
            ensemble_score = 0.0
            total_weight = 0.0
            
            # LLM score
            if 'llm' in scores and item_id in scores['llm']:
                llm_score = self._normalize_score(scores['llm'][item_id])
                ensemble_score += self.ensemble_weights['llm'] * llm_score
                total_weight += self.ensemble_weights['llm']
            
            # NCF score
            if 'ncf' in scores and item_id in scores['ncf']:
                ncf_score = self._normalize_score(scores['ncf'][item_id])
                ensemble_score += self.ensemble_weights['ncf'] * ncf_score
                total_weight += self.ensemble_weights['ncf']
            
            # LightGBM score
            if 'lightgbm' in scores and item_id in scores['lightgbm']:
                lgb_score = self._normalize_score(scores['lightgbm'][item_id])
                ensemble_score += self.ensemble_weights['lightgbm'] * lgb_score
                total_weight += self.ensemble_weights['lightgbm']
            
            # Normalize by total weight (in case some models are missing)
            if total_weight > 0:
                final_scores[item_id] = ensemble_score / total_weight
            else:
                final_scores[item_id] = 0.0
        
        return final_scores
    
    def _normalize_score(self, score: float, method: str = 'sigmoid') -> float:
        """
        Normalize score to [0, 1] range
        """
        if method == 'sigmoid':
            # Sigmoid normalization
            return 1.0 / (1.0 + np.exp(-score))
        elif method == 'minmax':
            # Min-max normalization (assumes score in reasonable range)
            return max(0.0, min(1.0, score))
        else:
            return score
    
    async def _enrich_recommendations(
        self,
        ranked_items: List[Tuple[int, float]],
        candidates: List[Dict[str, Any]],
        scores: Dict[str, Dict[int, float]],
        query: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Enrich recommendations with details and explanations
        """
        candidates_by_id = {c['item_id']: c for c in candidates}
        
        enriched = []
        
        for rank, (item_id, final_score) in enumerate(ranked_items, 1):
            candidate = candidates_by_id.get(item_id, {})
            
            # Build explanation
            explanation = self._build_explanation(
                item_id=item_id,
                candidate=candidate,
                scores=scores,
                query=query
            )
            
            enriched.append({
                'rank': rank,
                'item_id': item_id,
                'name': candidate.get('name', f'Item {item_id}'),
                'category': candidate.get('category', 'unknown'),
                'final_score': float(final_score),
                'component_scores': {
                    'llm': float(scores.get('llm', {}).get(item_id, 0.0)),
                    'ncf': float(scores.get('ncf', {}).get(item_id, 0.0)),
                    'lightgbm': float(scores.get('lightgbm', {}).get(item_id, 0.0))
                },
                'explanation': explanation,
                'metadata': candidate.get('metadata', {})
            })
        
        return enriched
    
    def _build_explanation(
        self,
        item_id: int,
        candidate: Dict[str, Any],
        scores: Dict[str, Dict[int, float]],
        query: str
    ) -> str:
        """
        Build human-readable explanation for recommendation
        """
        reasons = []
        
        # LLM explanation
        if candidate.get('llm_explanation'):
            reasons.append(f"✨ {candidate['llm_explanation']}")
        
        # NCF explanation
        if 'ncf' in scores and item_id in scores['ncf']:
            ncf_score = scores['ncf'][item_id]
            if ncf_score > 0.7:
                reasons.append("👥 Popular among users with similar tastes")
        
        # LightGBM explanation
        if 'lightgbm' in scores and item_id in scores['lightgbm']:
            lgb_score = scores['lightgbm'][item_id]
            if lgb_score > 0:
                reasons.append("⭐ Highly rated and well-reviewed")
        
        # Combine reasons
        if reasons:
            return " • ".join(reasons)
        else:
            return f"Matches your query: {query}"
    
    def update_statistics(
        self,
        user_stats: Optional[Dict[int, Dict]] = None,
        item_stats: Optional[Dict[int, Dict]] = None
    ):
        """
        Update cached statistics for LightGBM ranker
        """
        if self.has_lightgbm:
            self.lightgbm_service.batch_update_stats(
                user_stats=user_stats,
                item_stats=item_stats
            )
            logger.info("✅ Statistics cache updated")
    
    def get_service_info(self) -> Dict[str, Any]:
        """
        Get information about available services
        """
        return {
            'llm_enabled': self.use_llm,
            'ncf_available': self.has_ncf,
            'lightgbm_available': self.has_lightgbm,
            'ensemble_weights': self.ensemble_weights,
            'components': {
                'llm': 'LLM-based semantic matching and candidate generation',
                'ncf': 'Neural Collaborative Filtering for user-item interactions',
                'lightgbm': 'Gradient boosting for feature-based ranking'
            }
        }


# Example usage and testing
async def test_integrated_service():
    """
    Test the integrated recommendation service
    """
    print("="*80)
    print("🧪 Testing Integrated Recommendation Service")
    print("="*80)
    
    # Initialize service
    service = IntegratedRecommendationService(
        lightgbm_model_path='models/ranker/lightgbm_ranker.pkl',
        lightgbm_feature_stats_path='models/ranker/feature_stats.pkl',
        use_llm=True,
        ensemble_weights={'llm': 0.5, 'ncf': 0.2, 'lightgbm': 0.3}
    )
    
    # Print service info
    info = service.get_service_info()
    print("\n📊 Service Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test recommendation
    print("\n" + "="*80)
    print("🎯 Testing Recommendations")
    print("="*80)
    
    recommendations = await service.get_recommendations(
        user_id=42,
        query="Find me hidden gems for authentic Turkish breakfast",
        context={
            'location': {'lat': 41.0082, 'lon': 28.9784},
            'time': datetime.now(),
            'preferences': ['authentic', 'local', 'breakfast']
        },
        top_k=5
    )
    
    print(f"\n✅ Got {len(recommendations)} recommendations:")
    for rec in recommendations:
        print(f"\n{rec['rank']}. {rec['name']} (ID: {rec['item_id']})")
        print(f"   Category: {rec['category']}")
        print(f"   Final Score: {rec['final_score']:.4f}")
        print(f"   Component Scores:")
        for component, score in rec['component_scores'].items():
            print(f"     - {component}: {score:.4f}")
        print(f"   Explanation: {rec['explanation']}")
    
    print("\n✅ Test completed successfully!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_integrated_service())
