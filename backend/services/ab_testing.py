"""
A/B Testing Framework for Intent Classifier
Test new models before full deployment
"""

import hashlib
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class IntentClassifierABTest:
    """
    A/B test new models before full deployment
    
    Features:
    - Deterministic assignment (same session always gets same model)
    - Configurable traffic split
    - Statistical significance testing
    - Performance metrics comparison
    """
    
    def __init__(self,
                 test_percentage: float = 0.1,
                 results_file: str = "./ab_test_results.json",
                 min_samples_for_significance: int = 100):
        """
        Initialize A/B testing framework
        
        Args:
            test_percentage: Percentage of traffic for candidate model (0.0-1.0)
            results_file: File to store test results
            min_samples_for_significance: Minimum samples needed for stats test
        """
        self.test_percentage = test_percentage
        self.results_file = Path(results_file)
        self.min_samples_for_significance = min_samples_for_significance
        
        # Model references (would be actual model objects in production)
        self.current_model = None
        self.candidate_model = None
        
        # Results storage
        self.results = self._load_results()
    
    def _load_results(self) -> Dict[str, Any]:
        """Load A/B test results from file"""
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                return json.load(f)
        
        return {
            'test_start': None,
            'test_end': None,
            'current_model': {
                'classifications': [],
                'feedback': []
            },
            'candidate_model': {
                'classifications': [],
                'feedback': []
            }
        }
    
    def _save_results(self):
        """Save A/B test results to file"""
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def start_test(self, current_model_version: str, candidate_model_version: str):
        """
        Start a new A/B test
        
        Args:
            current_model_version: Version identifier for current model
            candidate_model_version: Version identifier for candidate model
        """
        
        logger.info("=" * 60)
        logger.info("Starting A/B Test")
        logger.info("=" * 60)
        logger.info(f"Current Model: {current_model_version}")
        logger.info(f"Candidate Model: {candidate_model_version}")
        logger.info(f"Traffic Split: {(1-self.test_percentage)*100:.0f}% / {self.test_percentage*100:.0f}%")
        logger.info("=" * 60 + "\n")
        
        self.results = {
            'test_start': datetime.utcnow().isoformat(),
            'test_end': None,
            'current_model_version': current_model_version,
            'candidate_model_version': candidate_model_version,
            'traffic_split': {
                'current': 1 - self.test_percentage,
                'candidate': self.test_percentage
            },
            'current_model': {
                'classifications': [],
                'feedback': []
            },
            'candidate_model': {
                'classifications': [],
                'feedback': []
            }
        }
        
        self._save_results()
    
    def classify_with_ab_test(self, query: str, session_id: str,
                              actual_classify_func: callable) -> Dict[str, Any]:
        """
        Route query to A or B model based on session
        
        Args:
            query: User query
            session_id: Session identifier
            actual_classify_func: Function to actually classify (receives model_version)
            
        Returns:
            Classification result with model_version added
        """
        
        # Determine which model to use
        use_candidate = self._should_use_candidate(session_id)
        model_version = 'candidate' if use_candidate else 'current'
        
        # Classify (in production, this would call the actual model)
        result = actual_classify_func(model_version)
        
        # Add model version to result
        result['model_version'] = model_version
        result['ab_test_active'] = True
        
        # Log classification
        self._log_classification(model_version, query, result, session_id)
        
        return result
    
    def _should_use_candidate(self, session_id: str) -> bool:
        """
        Determine if session should use candidate model
        
        Uses deterministic hash-based assignment so same session
        always gets same model
        """
        
        # Hash session ID to get consistent assignment
        hash_value = hashlib.md5(session_id.encode()).hexdigest()
        hash_int = int(hash_value, 16)
        
        # Convert to 0-1 range
        normalized = (hash_int % 10000) / 10000.0
        
        # Assign to candidate if below threshold
        return normalized < self.test_percentage
    
    def _log_classification(self, model_version: str, query: str,
                           result: Dict, session_id: str):
        """Log classification for analysis"""
        
        model_key = f'{model_version}_model'
        
        classification_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'session_id': session_id,
            'query': query,
            'intent': result.get('intent'),
            'confidence': result.get('confidence'),
            'latency_ms': result.get('latency_ms', 0)
        }
        
        self.results[model_key]['classifications'].append(classification_record)
        
        # Save periodically (every 10 classifications)
        if len(self.results[model_key]['classifications']) % 10 == 0:
            self._save_results()
    
    def log_feedback(self, session_id: str, is_correct: bool,
                    actual_intent: Optional[str] = None):
        """
        Log user feedback for the classification
        
        Args:
            session_id: Session that provided feedback
            is_correct: Whether classification was correct
            actual_intent: Correct intent if is_correct=False
        """
        
        # Find which model was used for this session
        use_candidate = self._should_use_candidate(session_id)
        model_key = 'candidate_model' if use_candidate else 'current_model'
        
        feedback_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'session_id': session_id,
            'is_correct': is_correct,
            'actual_intent': actual_intent
        }
        
        self.results[model_key]['feedback'].append(feedback_record)
        self._save_results()
    
    async def analyze_ab_test_results(self) -> Dict[str, Any]:
        """
        Analyze A/B test results and compare models
        
        Returns:
            Comprehensive comparison with recommendation
        """
        
        logger.info("\n" + "=" * 60)
        logger.info("A/B Test Results Analysis")
        logger.info("=" * 60)
        
        # Calculate metrics for each model
        current_metrics = self._calculate_metrics('current_model')
        candidate_metrics = self._calculate_metrics('candidate_model')
        
        # Compare metrics
        comparison = self._compare_metrics(current_metrics, candidate_metrics)
        
        # Statistical significance test
        is_significant = self._check_statistical_significance(
            current_metrics,
            candidate_metrics
        )
        
        # Determine winner
        winner = self._determine_winner(current_metrics, candidate_metrics, is_significant)
        
        # Generate recommendation
        recommendation = self._get_deployment_recommendation(
            winner,
            is_significant,
            current_metrics,
            candidate_metrics
        )
        
        result = {
            'test_duration_hours': self._get_test_duration_hours(),
            'current_model': current_metrics,
            'candidate_model': candidate_metrics,
            'comparison': comparison,
            'is_statistically_significant': is_significant,
            'winner': winner,
            'recommendation': recommendation,
            'analysis_timestamp': datetime.utcnow().isoformat()
        }
        
        self._print_analysis(result)
        
        return result
    
    def _calculate_metrics(self, model_key: str) -> Dict[str, Any]:
        """Calculate performance metrics for a model"""
        
        model_data = self.results[model_key]
        classifications = model_data['classifications']
        feedback = model_data['feedback']
        
        if not classifications:
            return {
                'sample_count': 0,
                'accuracy': None,
                'avg_latency_ms': None,
                'avg_confidence': None,
                'user_satisfaction': None
            }
        
        # Calculate accuracy from feedback
        accuracy = None
        if feedback:
            correct_count = sum(1 for f in feedback if f['is_correct'])
            accuracy = correct_count / len(feedback)
        
        # Calculate average latency
        latencies = [c['latency_ms'] for c in classifications if 'latency_ms' in c]
        avg_latency = sum(latencies) / len(latencies) if latencies else None
        
        # Calculate average confidence
        confidences = [c['confidence'] for c in classifications if c.get('confidence')]
        avg_confidence = sum(confidences) / len(confidences) if confidences else None
        
        # User satisfaction from feedback
        satisfaction = None
        if feedback:
            satisfaction = sum(1 for f in feedback if f['is_correct']) / len(feedback)
        
        return {
            'sample_count': len(classifications),
            'feedback_count': len(feedback),
            'accuracy': accuracy,
            'avg_latency_ms': avg_latency,
            'avg_confidence': avg_confidence,
            'user_satisfaction': satisfaction
        }
    
    def _compare_metrics(self, current: Dict, candidate: Dict) -> Dict[str, Any]:
        """Compare metrics between models"""
        
        comparison = {}
        
        for metric in ['accuracy', 'avg_latency_ms', 'avg_confidence', 'user_satisfaction']:
            current_val = current.get(metric)
            candidate_val = candidate.get(metric)
            
            if current_val is not None and candidate_val is not None:
                # For latency, lower is better
                if metric == 'avg_latency_ms':
                    improvement = current_val - candidate_val
                    better = 'candidate' if improvement > 0 else 'current'
                else:
                    improvement = candidate_val - current_val
                    better = 'candidate' if improvement > 0 else 'current'
                
                comparison[metric] = {
                    'current': current_val,
                    'candidate': candidate_val,
                    'improvement': improvement,
                    'improvement_pct': (improvement / current_val * 100) if current_val != 0 else 0,
                    'better': better
                }
        
        return comparison
    
    def _check_statistical_significance(self, current: Dict, candidate: Dict) -> bool:
        """
        Check if differences are statistically significant
        
        Simplified version - in production would use proper statistical tests
        (e.g., Chi-square test for accuracy, t-test for latency)
        """
        
        # Need minimum samples
        if current['sample_count'] < self.min_samples_for_significance:
            return False
        if candidate['sample_count'] < self.min_samples_for_significance:
            return False
        
        # Check if accuracy difference is meaningful
        if current['accuracy'] is not None and candidate['accuracy'] is not None:
            accuracy_diff = abs(current['accuracy'] - candidate['accuracy'])
            
            # Consider significant if difference > 5% and sufficient samples
            if accuracy_diff > 0.05:
                return True
        
        return False
    
    def _determine_winner(self, current: Dict, candidate: Dict,
                         is_significant: bool) -> str:
        """Determine which model performed better"""
        
        if not is_significant:
            return 'inconclusive'
        
        # Score based on multiple factors
        current_score = 0
        candidate_score = 0
        
        if current['accuracy'] is not None and candidate['accuracy'] is not None:
            if candidate['accuracy'] > current['accuracy']:
                candidate_score += 2
            elif current['accuracy'] > candidate['accuracy']:
                current_score += 2
        
        if current['avg_latency_ms'] is not None and candidate['avg_latency_ms'] is not None:
            if candidate['avg_latency_ms'] < current['avg_latency_ms']:
                candidate_score += 1
            elif current['avg_latency_ms'] < candidate['avg_latency_ms']:
                current_score += 1
        
        if current['user_satisfaction'] is not None and candidate['user_satisfaction'] is not None:
            if candidate['user_satisfaction'] > current['user_satisfaction']:
                candidate_score += 2
            elif current['user_satisfaction'] > candidate['user_satisfaction']:
                current_score += 2
        
        if candidate_score > current_score:
            return 'candidate'
        elif current_score > candidate_score:
            return 'current'
        else:
            return 'tie'
    
    def _get_deployment_recommendation(self, winner: str, is_significant: bool,
                                      current: Dict, candidate: Dict) -> Dict[str, str]:
        """Generate deployment recommendation"""
        
        if not is_significant:
            return {
                'action': 'continue_testing',
                'reason': f'Need more samples for statistical significance (current: {current["sample_count"]}, candidate: {candidate["sample_count"]})'
            }
        
        if winner == 'candidate':
            # Check if candidate is significantly better
            if candidate['accuracy'] and current['accuracy']:
                improvement = candidate['accuracy'] - current['accuracy']
                if improvement > 0.05:  # 5% improvement
                    return {
                        'action': 'deploy_candidate',
                        'reason': f'Candidate model shows {improvement:.1%} accuracy improvement'
                    }
            
            return {
                'action': 'deploy_candidate',
                'reason': 'Candidate model performs better overall'
            }
        
        elif winner == 'current':
            return {
                'action': 'keep_current',
                'reason': 'Current model still performs better'
            }
        
        else:
            return {
                'action': 'continue_testing',
                'reason': 'Models perform similarly, need more data to decide'
            }
    
    def _get_test_duration_hours(self) -> Optional[float]:
        """Calculate test duration in hours"""
        
        if not self.results.get('test_start'):
            return None
        
        start = datetime.fromisoformat(self.results['test_start'])
        end = datetime.utcnow()
        
        duration = end - start
        return duration.total_seconds() / 3600
    
    def _print_analysis(self, result: Dict):
        """Print analysis results"""
        
        duration = result.get('test_duration_hours')
        if duration:
            logger.info(f"\nTest Duration: {duration:.1f} hours")
        
        logger.info("\n📊 Current Model:")
        self._print_metrics(result['current_model'])
        
        logger.info("\n📊 Candidate Model:")
        self._print_metrics(result['candidate_model'])
        
        logger.info("\n🔍 Comparison:")
        for metric, data in result['comparison'].items():
            logger.info(f"  {metric}:")
            logger.info(f"    Current:   {data['current']:.4f}")
            logger.info(f"    Candidate: {data['candidate']:.4f}")
            logger.info(f"    Change:    {data['improvement']:+.4f} ({data['improvement_pct']:+.1f}%)")
            logger.info(f"    Better:    {data['better']}")
        
        logger.info(f"\n🏆 Winner: {result['winner'].upper()}")
        logger.info(f"📈 Statistically Significant: {'YES' if result['is_statistically_significant'] else 'NO'}")
        
        logger.info(f"\n💡 Recommendation:")
        logger.info(f"  Action: {result['recommendation']['action'].upper()}")
        logger.info(f"  Reason: {result['recommendation']['reason']}")
        
        logger.info("=" * 60 + "\n")
    
    def _print_metrics(self, metrics: Dict):
        """Print metrics for a model"""
        
        logger.info(f"  Samples: {metrics['sample_count']}")
        logger.info(f"  Feedback: {metrics['feedback_count']}")
        
        if metrics['accuracy'] is not None:
            logger.info(f"  Accuracy: {metrics['accuracy']:.2%}")
        
        if metrics['avg_latency_ms'] is not None:
            logger.info(f"  Avg Latency: {metrics['avg_latency_ms']:.1f}ms")
        
        if metrics['avg_confidence'] is not None:
            logger.info(f"  Avg Confidence: {metrics['avg_confidence']:.2%}")
        
        if metrics['user_satisfaction'] is not None:
            logger.info(f"  User Satisfaction: {metrics['user_satisfaction']:.2%}")
    
    def end_test(self):
        """End the A/B test"""
        
        self.results['test_end'] = datetime.utcnow().isoformat()
        self._save_results()
        
        logger.info("✅ A/B test ended")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("A/B Testing Framework - Demo")
    print("=" * 60)
    
    # Create A/B test
    ab_test = IntentClassifierABTest(
        test_percentage=0.3,  # 30% to candidate
        results_file="./test_ab_results.json"
    )
    
    # Start test
    ab_test.start_test("v1.0", "v2.0")
    
    # Simulate some classifications
    def mock_classify(model_version):
        import random
        # Candidate model is slightly better
        base_accuracy = 0.85 if model_version == 'current' else 0.88
        
        return {
            'intent': 'restaurant',
            'confidence': base_accuracy + random.uniform(-0.1, 0.1),
            'latency_ms': 15 + random.uniform(-5, 5)
        }
    
    # Simulate 200 classifications
    for i in range(200):
        session_id = f"session_{i % 50}"  # 50 unique sessions
        result = ab_test.classify_with_ab_test(
            f"test query {i}",
            session_id,
            mock_classify
        )
    
    # Simulate some feedback
    import random
    for i in range(50):
        session_id = f"session_{i}"
        is_correct = random.random() < 0.87  # 87% accuracy
        ab_test.log_feedback(session_id, is_correct)
    
    # Analyze results
    import asyncio
    analysis = asyncio.run(ab_test.analyze_ab_test_results())
    
    print(f"\n✅ Recommendation: {analysis['recommendation']['action']}")
    print(f"   Reason: {analysis['recommendation']['reason']}")
