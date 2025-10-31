"""
Ranking Experiments - A/B Tests for Neural vs Keyword Ranking

This module provides pre-configured experiments for testing
different ranking strategies and configurations.

Author: Istanbul AI Team
Date: October 31, 2025
"""

import logging
from typing import Dict, List, Any, Optional
from .ab_test_framework import (
    ABTestFramework,
    Experiment,
    Variant,
    VariantStatus
)

logger = logging.getLogger(__name__)


class RankingExperiments:
    """
    Pre-configured ranking experiments
    
    Provides experiments for testing:
    - Neural vs Keyword ranking
    - Ranking weight configurations
    - Confidence thresholds
    """
    
    def __init__(self, ab_framework: ABTestFramework):
        """
        Initialize ranking experiments
        
        Args:
            ab_framework: A/B test framework instance
        """
        self.ab_framework = ab_framework
        self._create_experiments()
    
    def _create_experiments(self):
        """Create all ranking experiments"""
        self._create_neural_vs_keyword_experiment()
        self._create_weight_tuning_experiment()
        self._create_threshold_experiment()
        self._create_finetuned_model_experiment()  # NEW: Fine-tuned model experiment
    
    def _create_neural_vs_keyword_experiment(self):
        """
        Experiment: Neural vs Keyword Ranking
        
        Tests whether neural semantic ranking provides
        better results than simple keyword/rating ranking.
        """
        experiment = Experiment(
            id="ranking_method",
            name="Neural vs Keyword Ranking",
            description="Compare neural semantic ranking against keyword-based ranking",
            variants=[
                Variant(
                    id="neural",
                    name="Neural Ranking",
                    weight=0.45,  # 45% traffic
                    config={
                        "use_neural_ranking": True,
                        "semantic_weight": 0.60,
                        "context_weight": 0.20,
                        "popularity_weight": 0.10,
                        "recency_weight": 0.10
                    }
                ),
                Variant(
                    id="keyword",
                    name="Keyword Ranking",
                    weight=0.45,  # 45% traffic
                    config={
                        "use_neural_ranking": False,
                        "rank_by": "rating"  # Simple rating-based ranking
                    }
                ),
                Variant(
                    id="control",
                    name="Control (Current System)",
                    weight=0.10,  # 10% control group
                    config={
                        "use_neural_ranking": True,  # Current default
                        "semantic_weight": 0.60
                    }
                )
            ],
            metrics=[
                "response_time",
                "user_satisfaction",
                "click_through_rate",
                "query_refinement_rate",
                "result_relevance"
            ],
            duration_days=7,
            min_sample_size=1000
        )
        
        self.ab_framework.create_experiment(experiment)
        logger.info("✅ Created experiment: Neural vs Keyword Ranking")
    
    def _create_weight_tuning_experiment(self):
        """
        Experiment: Ranking Weight Optimization
        
        Tests different weight configurations for multi-factor ranking.
        """
        experiment = Experiment(
            id="ranking_weights",
            name="Ranking Weight Optimization",
            description="Find optimal balance between semantic, context, and other factors",
            variants=[
                Variant(
                    id="high_semantic",
                    name="High Semantic (70%)",
                    weight=0.33,
                    config={
                        "use_neural_ranking": True,
                        "semantic_weight": 0.70,
                        "context_weight": 0.15,
                        "popularity_weight": 0.10,
                        "recency_weight": 0.05
                    }
                ),
                Variant(
                    id="balanced",
                    name="Balanced (60/20/20)",
                    weight=0.34,
                    config={
                        "use_neural_ranking": True,
                        "semantic_weight": 0.60,
                        "context_weight": 0.20,
                        "popularity_weight": 0.10,
                        "recency_weight": 0.10
                    }
                ),
                Variant(
                    id="high_context",
                    name="High Context (50/30/20)",
                    weight=0.33,
                    config={
                        "use_neural_ranking": True,
                        "semantic_weight": 0.50,
                        "context_weight": 0.30,
                        "popularity_weight": 0.15,
                        "recency_weight": 0.05
                    }
                )
            ],
            metrics=[
                "result_relevance",
                "personalization_score",
                "user_satisfaction"
            ],
            duration_days=7,
            min_sample_size=1500
        )
        
        self.ab_framework.create_experiment(experiment)
        logger.info("✅ Created experiment: Ranking Weight Optimization")
    
    def _create_threshold_experiment(self):
        """
        Experiment: Neural Classifier Confidence Threshold
        
        Tests different confidence thresholds for hybrid classification.
        """
        experiment = Experiment(
            id="confidence_threshold",
            name="Neural Classifier Threshold Optimization",
            description="Find optimal confidence threshold for neural vs keyword fallback",
            variants=[
                Variant(
                    id="threshold_065",
                    name="Threshold 0.65 (Aggressive)",
                    weight=0.33,
                    config={
                        "neural_confidence_threshold": 0.65,
                        "use_neural_ranking": True
                    }
                ),
                Variant(
                    id="threshold_070",
                    name="Threshold 0.70 (Current)",
                    weight=0.34,
                    config={
                        "neural_confidence_threshold": 0.70,
                        "use_neural_ranking": True
                    }
                ),
                Variant(
                    id="threshold_075",
                    name="Threshold 0.75 (Conservative)",
                    weight=0.33,
                    config={
                        "neural_confidence_threshold": 0.75,
                        "use_neural_ranking": True
                    }
                )
            ],
            metrics=[
                "neural_usage_rate",
                "classification_accuracy",
                "fallback_rate",
                "user_satisfaction"
            ],
            duration_days=7,
            min_sample_size=2000
        )
        
        self.ab_framework.create_experiment(experiment)
        logger.info("✅ Created experiment: Confidence Threshold Optimization")
    
    def _create_finetuned_model_experiment(self):
        """
        Experiment: Base vs Fine-tuned Intent Classifier
        
        Tests whether the fine-tuned Istanbul-specific model provides
        better intent classification than the base DistilBERT model.
        """
        from datetime import datetime, timedelta
        
        experiment = Experiment(
            id="finetuned_model_comparison",
            name="Base vs Fine-tuned Intent Classifier",
            description="Compare base DistilBERT vs Istanbul fine-tuned model for intent classification",
            variants=[
                Variant(
                    id="finetuned",
                    name="Fine-tuned Istanbul Model",
                    description="DistilBERT fine-tuned on 7,202 Istanbul-specific examples",
                    traffic_percentage=0.70,  # 70% of users get fine-tuned model
                    config={
                        'use_finetuned_model': True,
                        'confidence_threshold': 0.75,
                        'model_type': 'finetuned',
                        'expected_accuracy': 0.93
                    }
                ),
                Variant(
                    id="base",
                    name="Base DistilBERT Model",
                    description="Pre-trained DistilBERT without Istanbul fine-tuning",
                    traffic_percentage=0.20,  # 20% of users get base model
                    config={
                        'use_finetuned_model': False,
                        'confidence_threshold': 0.70,
                        'model_type': 'base',
                        'expected_accuracy': 0.78
                    }
                ),
                Variant(
                    id="control",
                    name="Control Group",
                    description="Control group for statistical analysis",
                    traffic_percentage=0.10,  # 10% control
                    config={
                        'use_finetuned_model': True,  # Same as primary
                        'confidence_threshold': 0.75,
                        'model_type': 'control'
                    }
                )
            ],
            metrics=[
                'intent_classification_accuracy',
                'confidence_score',
                'response_latency',
                'user_satisfaction',
                'query_success_rate',
                'fallback_rate',
                'multi_turn_queries'
            ],
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=14),  # 2 week test
            status='active',
            min_sample_size=500,  # Minimum samples per variant
            success_metric='intent_classification_accuracy',
            success_threshold=0.05  # 5% improvement threshold
        )
        
        self.ab_framework.create_experiment(experiment)
        logger.info("✅ Created fine-tuned model experiment: Base vs Fine-tuned Intent Classifier")
    
    def execute_ranking(
        self,
        experiment_id: str,
        user_id: str,
        query: str,
        results: List[Dict],
        ranker,  # NeuralResponseRanker instance
        user_profile: Optional[Any] = None
    ) -> List[Dict]:
        """
        Execute ranking with A/B test variant
        
        Args:
            experiment_id: Experiment to run
            user_id: User identifier
            query: Search query
            results: Results to rank
            ranker: NeuralResponseRanker instance
            user_profile: Optional user profile
            
        Returns:
            Ranked results according to assigned variant
        """
        def rank_with_config(config: Dict[str, Any]) -> List[Dict]:
            """Rank results using variant configuration"""
            use_neural = config.get('use_neural_ranking', True)
            
            if not use_neural:
                # Keyword ranking: sort by rating
                rank_by = config.get('rank_by', 'rating')
                return sorted(
                    results,
                    key=lambda x: x.get(rank_by, x.get('popularity', 0)),
                    reverse=True
                )
            
            # Neural ranking with custom weights
            from ..routing.neural_response_ranker import RankingConfig
            
            ranking_config = RankingConfig(
                semantic_weight=config.get('semantic_weight', 0.60),
                context_weight=config.get('context_weight', 0.20),
                popularity_weight=config.get('popularity_weight', 0.10),
                recency_weight=config.get('recency_weight', 0.10)
            )
            
            # Temporarily update ranker config
            original_config = ranker.config
            ranker.config = ranking_config
            
            try:
                # Build user context
                user_context = None
                if user_profile:
                    user_context = {
                        'preferences': user_profile.preferences,
                        'history': user_profile.history
                    }
                
                # Rank using neural ranker
                ranking_result = ranker.rank_results(
                    query=query,
                    results=results,
                    user_context=user_context
                )
                
                return ranking_result.ranked_results if hasattr(ranking_result, 'ranked_results') else ranking_result
                
            finally:
                # Restore original config
                ranker.config = original_config
        
        def extract_metrics(ranked_results: List[Dict]) -> Dict[str, Any]:
            """Extract metrics from ranking result"""
            if not ranked_results:
                return {}
            
            # Calculate average semantic similarity if available
            similarities = [r.get('semantic_similarity', 0) for r in ranked_results]
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0
            
            return {
                'avg_semantic_similarity': avg_similarity,
                'num_results': len(ranked_results),
                'top_result_score': ranked_results[0].get('neural_score', ranked_results[0].get('rating', 0))
            }
        
        # Execute with A/B test framework
        return self.ab_framework.execute_variant(
            experiment_id=experiment_id,
            user_id=user_id,
            execution_fn=rank_with_config,
            metrics_fn=extract_metrics
        )
    
    def log_user_click(
        self,
        experiment_id: str,
        user_id: str,
        clicked_position: int,
        result_id: str
    ):
        """
        Log user click on ranking result
        
        Args:
            experiment_id: Experiment identifier
            user_id: User identifier
            clicked_position: Position in ranking (0-indexed)
            result_id: ID of clicked result
        """
        self.ab_framework.log_user_interaction(
            experiment_id=experiment_id,
            user_id=user_id,
            event_type='result_click',
            metrics={
                'clicked_position': clicked_position,
                'result_id': result_id,
                'is_top_3': clicked_position < 3
            }
        )
    
    def log_user_feedback(
        self,
        experiment_id: str,
        user_id: str,
        satisfaction_score: int,  # 1-5
        relevance_score: Optional[int] = None
    ):
        """
        Log user feedback on results
        
        Args:
            experiment_id: Experiment identifier
            user_id: User identifier
            satisfaction_score: Overall satisfaction (1-5)
            relevance_score: Result relevance (1-5)
        """
        metrics = {
            'satisfaction_score': satisfaction_score,
            'is_satisfied': satisfaction_score >= 4
        }
        
        if relevance_score is not None:
            metrics['relevance_score'] = relevance_score
            metrics['is_relevant'] = relevance_score >= 4
        
        self.ab_framework.log_user_interaction(
            experiment_id=experiment_id,
            user_id=user_id,
            event_type='user_feedback',
            metrics=metrics
        )
    
    def get_experiment_stats(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get statistics for ranking experiment
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Statistics dictionary
        """
        return self.ab_framework.get_stats(experiment_id)


def create_ranking_experiments(ab_framework: ABTestFramework) -> RankingExperiments:
    """
    Factory function to create ranking experiments
    
    Args:
        ab_framework: A/B test framework instance
        
    Returns:
        RankingExperiments instance
    """
    return RankingExperiments(ab_framework)
