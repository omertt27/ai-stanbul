"""
NCF Model Validator

Validates new models before deployment to production.

Validation criteria:
- Accuracy on hold-out test set
- NDCG (Normalized Discounted Cumulative Gain)
- Diversity of recommendations
- Inference latency
- Comparison against current production model

Author: AI Istanbul Team
Date: February 10, 2026
"""

import logging
import time
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import torch
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelValidator:
    """
    Validates NCF models before production deployment.
    """
    
    def __init__(
        self,
        min_accuracy: float = 0.60,
        max_accuracy_drop: float = 0.02,
        min_ndcg: float = 0.30,
        min_diversity: float = 0.30,
        max_latency_increase: float = 1.2
    ):
        """
        Initialize model validator.
        
        Args:
            min_accuracy: Minimum acceptable accuracy
            max_accuracy_drop: Maximum allowed accuracy drop vs current model
            min_ndcg: Minimum NDCG@10 score
            min_diversity: Minimum recommendation diversity
            max_latency_increase: Maximum latency increase ratio
        """
        self.min_accuracy = min_accuracy
        self.max_accuracy_drop = max_accuracy_drop
        self.min_ndcg = min_ndcg
        self.min_diversity = min_diversity
        self.max_latency_increase = max_latency_increase
    
    def calculate_accuracy(
        self,
        model,
        test_data: np.ndarray,
        top_k: int = 10
    ) -> float:
        """
        Calculate recommendation accuracy on test set.
        
        Args:
            model: Model to evaluate (with predict method)
            test_data: Test data array (user_idx, item_idx, rating)
            top_k: Number of recommendations to consider
            
        Returns:
            Accuracy score (0-1)
        """
        logger.info(f"📊 Calculating accuracy with top_k={top_k}...")
        
        hits = 0
        total = 0
        
        # Group by user
        user_interactions = {}
        for user_idx, item_idx, rating in test_data:
            if user_idx not in user_interactions:
                user_interactions[user_idx] = []
            user_interactions[user_idx].append((item_idx, rating))
        
        for user_idx, items in user_interactions.items():
            try:
                # Get top-k recommendations
                recommendations = model.predict(int(user_idx), top_k=top_k)
                rec_items = [item_idx for item_idx, _ in recommendations]
                
                # Check if any test items are in recommendations
                test_items = [item_idx for item_idx, rating in items if rating >= 1.0]
                
                for test_item in test_items:
                    total += 1
                    if test_item in rec_items:
                        hits += 1
                        
            except Exception as e:
                logger.warning(f"⚠️ Failed to get predictions for user {user_idx}: {e}")
                continue
        
        accuracy = hits / total if total > 0 else 0.0
        logger.info(f"✅ Accuracy: {accuracy:.4f} ({hits}/{total} hits)")
        
        return accuracy
    
    def calculate_ndcg(
        self,
        model,
        test_data: np.ndarray,
        k: int = 10
    ) -> float:
        """
        Calculate NDCG@k (Normalized Discounted Cumulative Gain).
        
        Args:
            model: Model to evaluate
            test_data: Test data
            k: Cutoff for NDCG calculation
            
        Returns:
            NDCG@k score (0-1)
        """
        logger.info(f"📊 Calculating NDCG@{k}...")
        
        ndcg_scores = []
        
        # Group by user
        user_interactions = {}
        for user_idx, item_idx, rating in test_data:
            if user_idx not in user_interactions:
                user_interactions[user_idx] = []
            user_interactions[user_idx].append((item_idx, rating))
        
        for user_idx, items in user_interactions.items():
            try:
                # Get recommendations
                recommendations = model.predict(int(user_idx), top_k=k)
                
                # Calculate DCG
                dcg = 0.0
                for i, (item_idx, _) in enumerate(recommendations[:k]):
                    # Find relevance (rating) for this item
                    relevance = 0.0
                    for test_item, rating in items:
                        if test_item == item_idx:
                            relevance = rating
                            break
                    
                    # DCG formula: sum(rel_i / log2(i+2))
                    dcg += relevance / np.log2(i + 2)
                
                # Calculate IDCG (ideal DCG)
                ideal_ratings = sorted([rating for _, rating in items], reverse=True)[:k]
                idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_ratings))
                
                # NDCG = DCG / IDCG
                if idcg > 0:
                    ndcg_scores.append(dcg / idcg)
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed NDCG calculation for user {user_idx}: {e}")
                continue
        
        avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
        logger.info(f"✅ NDCG@{k}: {avg_ndcg:.4f}")
        
        return float(avg_ndcg)
    
    def calculate_diversity(
        self,
        model,
        user_sample: List[int],
        top_k: int = 10
    ) -> float:
        """
        Calculate recommendation diversity (intra-list diversity).
        
        Measures how different recommended items are from each other.
        
        Args:
            model: Model to evaluate
            user_sample: Sample of user IDs to test
            top_k: Number of recommendations
            
        Returns:
            Diversity score (0-1)
        """
        logger.info("📊 Calculating recommendation diversity...")
        
        all_recommendations = set()
        diversity_scores = []
        
        for user_idx in user_sample[:100]:  # Sample 100 users
            try:
                recommendations = model.predict(user_idx, top_k=top_k)
                rec_items = [item_idx for item_idx, _ in recommendations]
                
                # Track unique items
                all_recommendations.update(rec_items)
                
                # Intra-list diversity: % unique items
                diversity = len(set(rec_items)) / len(rec_items)
                diversity_scores.append(diversity)
                
            except Exception:
                continue
        
        avg_diversity = np.mean(diversity_scores) if diversity_scores else 0.0
        catalog_coverage = len(all_recommendations)
        
        logger.info(f"✅ Diversity: {avg_diversity:.4f}")
        logger.info(f"✅ Catalog coverage: {catalog_coverage} unique items")
        
        return float(avg_diversity)
    
    def measure_latency(
        self,
        model,
        num_samples: int = 100
    ) -> Dict[str, float]:
        """
        Measure model inference latency.
        
        Args:
            model: Model to benchmark
            num_samples: Number of predictions to time
            
        Returns:
            Dictionary with latency statistics
        """
        logger.info(f"⏱️ Measuring inference latency ({num_samples} samples)...")
        
        latencies = []
        
        for i in range(num_samples):
            user_idx = i % 1000  # Cycle through users
            
            start = time.perf_counter()
            try:
                _ = model.predict(user_idx, top_k=10)
                latency_ms = (time.perf_counter() - start) * 1000
                latencies.append(latency_ms)
            except Exception:
                continue
        
        if not latencies:
            return {
                'p50_ms': 0.0,
                'p95_ms': 0.0,
                'p99_ms': 0.0,
                'avg_ms': 0.0
            }
        
        latencies = sorted(latencies)
        stats = {
            'p50_ms': latencies[int(len(latencies) * 0.5)],
            'p95_ms': latencies[int(len(latencies) * 0.95)],
            'p99_ms': latencies[int(len(latencies) * 0.99)],
            'avg_ms': np.mean(latencies)
        }
        
        logger.info(f"✅ Latency stats:")
        logger.info(f"  P50: {stats['p50_ms']:.2f}ms")
        logger.info(f"  P95: {stats['p95_ms']:.2f}ms")
        logger.info(f"  P99: {stats['p99_ms']:.2f}ms")
        
        return stats
    
    def validate_model(
        self,
        new_model,
        test_data: np.ndarray,
        current_model = None,
        current_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Complete validation of a new model.
        
        Args:
            new_model: New model to validate
            test_data: Test dataset
            current_model: Current production model (optional)
            current_metrics: Current model metrics (optional)
            
        Returns:
            Validation report with pass/fail status
        """
        logger.info("🔍 Starting model validation...")
        
        # Evaluate new model
        new_accuracy = self.calculate_accuracy(new_model, test_data)
        new_ndcg = self.calculate_ndcg(new_model, test_data)
        new_diversity = self.calculate_diversity(
            new_model,
            user_sample=list(range(100))
        )
        new_latency = self.measure_latency(new_model)
        
        # Build report
        report = {
            'new_model': {
                'accuracy': new_accuracy,
                'ndcg@10': new_ndcg,
                'diversity': new_diversity,
                'latency_p95_ms': new_latency['p95_ms']
            },
            'checks': {},
            'passed': False,
            'timestamp': time.time()
        }
        
        # Validation checks
        checks = {
            'min_accuracy': new_accuracy >= self.min_accuracy,
            'min_ndcg': new_ndcg >= self.min_ndcg,
            'min_diversity': new_diversity >= self.min_diversity
        }
        
        # Compare against current model if available
        if current_model or current_metrics:
            if current_metrics:
                current_accuracy = current_metrics.get('accuracy', 0.65)
                current_latency_p95 = current_metrics.get('latency_p95_ms', 50.0)
            else:
                current_accuracy = self.calculate_accuracy(current_model, test_data)
                current_latency = self.measure_latency(current_model)
                current_latency_p95 = current_latency['p95_ms']
            
            report['current_model'] = {
                'accuracy': current_accuracy,
                'latency_p95_ms': current_latency_p95
            }
            
            # Additional checks
            accuracy_drop = current_accuracy - new_accuracy
            latency_ratio = new_latency['p95_ms'] / current_latency_p95 if current_latency_p95 > 0 else 1.0
            
            checks['max_accuracy_drop'] = accuracy_drop <= self.max_accuracy_drop
            checks['max_latency_increase'] = latency_ratio <= self.max_latency_increase
            
            report['comparison'] = {
                'accuracy_change': new_accuracy - current_accuracy,
                'latency_ratio': latency_ratio
            }
        
        report['checks'] = checks
        report['passed'] = all(checks.values())
        
        # Log results
        logger.info("📋 Validation Report:")
        for metric, value in report['new_model'].items():
            logger.info(f"  {metric}: {value:.4f}")
        
        logger.info("✅ Validation Checks:")
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            logger.info(f"  {status} {check}: {passed}")
        
        if report['passed']:
            logger.info("🎉 Model validation PASSED!")
        else:
            logger.warning("⚠️ Model validation FAILED!")
        
        return report


def get_model_validator(**kwargs) -> ModelValidator:
    """
    Factory function to get model validator.
    
    Args:
        **kwargs: Validation threshold overrides
        
    Returns:
        ModelValidator instance
    """
    return ModelValidator(**kwargs)
