"""
A/B Testing System for LLM Experimentation
Allows testing different LLM configurations, prompts, and models

Updated: December 2024 - Using improved standardized prompt templates
"""
import hashlib
import json
import random
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
import redis
import logging

# Import improved prompt templates for A/B testing
from IMPROVED_PROMPT_TEMPLATES import IMPROVED_BASE_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class Experiment:
    """A/B test experiment configuration"""
    id: str
    name: str
    description: str
    variants: List[Dict[str, Any]]
    traffic_split: List[float]  # Must sum to 1.0
    start_date: str
    end_date: Optional[str]
    is_active: bool
    metrics: Dict[str, Any]
    
    def to_dict(self):
        return asdict(self)


class ABTestingManager:
    """Manage A/B testing experiments for LLM responses"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.experiments_key = "ab_tests:experiments"
        self.assignments_key = "ab_tests:assignments"
        self.metrics_key = "ab_tests:metrics"
    
    def create_experiment(
        self,
        name: str,
        description: str,
        variants: List[Dict[str, Any]],
        traffic_split: List[float] = None
    ) -> Experiment:
        """
        Create a new A/B test experiment
        
        Args:
            name: Experiment name
            description: What you're testing
            variants: List of variant configurations
            traffic_split: Traffic distribution (default: equal split)
        
        Returns:
            Created experiment
        
        Example:
            variants = [
                {
                    "name": "control",
                    "model": "llama-3.1-8b",
                    "temperature": 0.7,
                    "max_tokens": 512
                },
                {
                    "name": "higher_temp",
                    "model": "llama-3.1-8b",
                    "temperature": 0.9,
                    "max_tokens": 512
                }
            ]
        """
        # Validate traffic split
        if traffic_split is None:
            traffic_split = [1.0 / len(variants)] * len(variants)
        
        if len(traffic_split) != len(variants):
            raise ValueError("Traffic split must match number of variants")
        
        if abs(sum(traffic_split) - 1.0) > 0.001:
            raise ValueError("Traffic split must sum to 1.0")
        
        # Create experiment
        experiment_id = hashlib.md5(
            f"{name}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        
        experiment = Experiment(
            id=experiment_id,
            name=name,
            description=description,
            variants=variants,
            traffic_split=traffic_split,
            start_date=datetime.now().isoformat(),
            end_date=None,
            is_active=True,
            metrics={
                variant['name']: {
                    'requests': 0,
                    'positive_feedback': 0,
                    'negative_feedback': 0,
                    'avg_response_time': 0.0,
                    'total_tokens': 0
                }
                for variant in variants
            }
        )
        
        # Store in Redis
        self.redis.hset(
            self.experiments_key,
            experiment_id,
            json.dumps(experiment.to_dict())
        )
        
        logger.info(f"Created experiment: {name} (ID: {experiment_id})")
        return experiment
    
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get experiment by ID"""
        data = self.redis.hget(self.experiments_key, experiment_id)
        if data:
            return Experiment(**json.loads(data))
        return None
    
    def list_experiments(self, active_only: bool = False) -> List[Experiment]:
        """List all experiments"""
        experiments = []
        for exp_data in self.redis.hvals(self.experiments_key):
            exp = Experiment(**json.loads(exp_data))
            if not active_only or exp.is_active:
                experiments.append(exp)
        return experiments
    
    def assign_variant(
        self,
        experiment_id: str,
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Assign user to a variant (consistent assignment)
        
        Args:
            experiment_id: ID of the experiment
            user_id: User identifier (can be session ID, IP, etc.)
        
        Returns:
            Variant configuration or None if experiment not found
        """
        # Check if user already assigned
        assignment_key = f"{self.assignments_key}:{experiment_id}:{user_id}"
        existing = self.redis.get(assignment_key)
        
        if existing:
            return json.loads(existing)
        
        # Get experiment
        experiment = self.get_experiment(experiment_id)
        if not experiment or not experiment.is_active:
            return None
        
        # Deterministic assignment based on user_id hash
        hash_value = int(hashlib.md5(
            f"{experiment_id}{user_id}".encode()
        ).hexdigest(), 16)
        
        random.seed(hash_value)
        variant = random.choices(
            experiment.variants,
            weights=experiment.traffic_split
        )[0]
        
        # Store assignment (expires after 7 days)
        self.redis.setex(
            assignment_key,
            604800,  # 7 days
            json.dumps(variant)
        )
        
        logger.info(
            f"Assigned user {user_id} to variant '{variant['name']}' "
            f"in experiment {experiment_id}"
        )
        
        return variant
    
    def record_metric(
        self,
        experiment_id: str,
        variant_name: str,
        metric_type: str,
        value: float = 1.0
    ):
        """
        Record a metric for a variant
        
        Args:
            experiment_id: Experiment ID
            variant_name: Name of the variant
            metric_type: Type of metric ('request', 'positive_feedback', etc.)
            value: Metric value
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return
        
        # Update metrics
        if variant_name in experiment.metrics:
            if metric_type == 'request':
                experiment.metrics[variant_name]['requests'] += 1
            elif metric_type == 'positive_feedback':
                experiment.metrics[variant_name]['positive_feedback'] += 1
            elif metric_type == 'negative_feedback':
                experiment.metrics[variant_name]['negative_feedback'] += 1
            elif metric_type == 'response_time':
                current_avg = experiment.metrics[variant_name]['avg_response_time']
                current_count = experiment.metrics[variant_name]['requests']
                new_avg = (current_avg * current_count + value) / (current_count + 1)
                experiment.metrics[variant_name]['avg_response_time'] = new_avg
            elif metric_type == 'tokens':
                experiment.metrics[variant_name]['total_tokens'] += int(value)
        
        # Save updated experiment
        self.redis.hset(
            self.experiments_key,
            experiment_id,
            json.dumps(experiment.to_dict())
        )
    
    def get_results(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get experiment results and statistics
        
        Returns:
            Results with statistical analysis or None if experiment not found
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return None
        
        results = {
            'experiment': {
                'id': experiment.id,
                'name': experiment.name,
                'description': experiment.description,
                'start_date': experiment.start_date,
                'end_date': experiment.end_date,
                'is_active': experiment.is_active
            },
            'variants': []
        }
        
        for variant_name, metrics in experiment.metrics.items():
            total_feedback = (
                metrics['positive_feedback'] + metrics['negative_feedback']
            )
            
            satisfaction_rate = (
                metrics['positive_feedback'] / total_feedback * 100
                if total_feedback > 0 else 0
            )
            
            variant_result = {
                'name': variant_name,
                'requests': metrics['requests'],
                'positive_feedback': metrics['positive_feedback'],
                'negative_feedback': metrics['negative_feedback'],
                'satisfaction_rate': round(satisfaction_rate, 2),
                'avg_response_time': round(metrics['avg_response_time'], 3),
                'total_tokens': metrics['total_tokens'],
                'avg_tokens_per_request': (
                    metrics['total_tokens'] / metrics['requests']
                    if metrics['requests'] > 0 else 0
                )
            }
            
            results['variants'].append(variant_result)
        
        # Calculate winner
        if results['variants']:
            winner = max(
                results['variants'],
                key=lambda x: x['satisfaction_rate']
            )
            results['winner'] = winner['name']
        
        return results
    
    def stop_experiment(self, experiment_id: str):
        """Stop an experiment"""
        experiment = self.get_experiment(experiment_id)
        if experiment:
            experiment.is_active = False
            experiment.end_date = datetime.now().isoformat()
            
            self.redis.hset(
                self.experiments_key,
                experiment_id,
                json.dumps(experiment.to_dict())
            )
            
            logger.info(f"Stopped experiment: {experiment_id}")
    
    def delete_experiment(self, experiment_id: str):
        """Delete an experiment"""
        self.redis.hdel(self.experiments_key, experiment_id)
        
        # Clean up assignments
        pattern = f"{self.assignments_key}:{experiment_id}:*"
        for key in self.redis.scan_iter(match=pattern):
            self.redis.delete(key)
        
        logger.info(f"Deleted experiment: {experiment_id}")


# Example usage
def example_usage():
    """Example of how to use the A/B testing system"""
    import redis as redis_sync
    
    # Initialize Redis
    redis_client = redis_sync.from_url("redis://localhost:6379")
    
    # Create A/B testing manager
    ab_manager = ABTestingManager(redis_client)
    
    # Create experiment
    variants = [
        {
            "name": "control",
            "model": "llama-3.1-8b",
            "temperature": 0.7,
            "max_tokens": 512,
            "system_prompt": IMPROVED_BASE_PROMPT  # Use improved standardized prompt
        },
        {
            "name": "creative",
            "model": "llama-3.1-8b",
            "temperature": 0.9,  # Test higher creativity
            "max_tokens": 512,
            "system_prompt": IMPROVED_BASE_PROMPT  # Same base prompt, different temperature
        },
        {
            "name": "concise",
            "model": "llama-3.1-8b",
            "temperature": 0.7,
            "max_tokens": 300,  # Test shorter responses
            "system_prompt": IMPROVED_BASE_PROMPT  # Same base prompt, fewer tokens
        }
    ]
    
    experiment = ab_manager.create_experiment(
        name="Temperature Test",
        description="Testing different temperature settings",
        variants=variants,
        traffic_split=[0.5, 0.5]
    )
    
    print(f"Created experiment: {experiment.id}")
    
    # Assign variants to users
    user1_variant = ab_manager.assign_variant(experiment.id, "user123")
    print(f"User1 variant: {user1_variant['name']}")
    
    user2_variant = ab_manager.assign_variant(experiment.id, "user456")
    print(f"User2 variant: {user2_variant['name']}")
    
    # Record metrics
    ab_manager.record_metric(experiment.id, "control", "request")
    ab_manager.record_metric(experiment.id, "control", "positive_feedback")
    ab_manager.record_metric(experiment.id, "control", "response_time", 2.5)
    
    # Get results
    results = ab_manager.get_results(experiment.id)
    print(f"Results: {json.dumps(results, indent=2)}")


if __name__ == "__main__":
    example_usage()
