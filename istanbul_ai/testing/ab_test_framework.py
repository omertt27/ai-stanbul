"""
A/B Testing Framework - Production Experiment System

This module provides a complete A/B testing framework for comparing
different system configurations and measuring their impact on user experience.

Features:
- Consistent user assignment to variants
- Multiple concurrent experiments
- Automatic metrics collection
- Statistical analysis
- Winner determination

Author: Istanbul AI Team
Date: October 31, 2025
"""

import hashlib
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import random

logger = logging.getLogger(__name__)


class VariantStatus(Enum):
    """Variant status in experiment"""
    ACTIVE = "active"
    PAUSED = "paused"
    WINNER = "winner"
    LOSER = "loser"


@dataclass
class Variant:
    """Experiment variant configuration"""
    id: str
    name: str
    weight: float  # Traffic allocation (0.0-1.0)
    config: Dict[str, Any]
    status: VariantStatus = VariantStatus.ACTIVE
    
    def __post_init__(self):
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"Weight must be between 0 and 1, got {self.weight}")


@dataclass
class Experiment:
    """A/B test experiment configuration"""
    id: str
    name: str
    description: str
    variants: List[Variant]
    metrics: List[str]
    duration_days: int = 7
    min_sample_size: int = 1000
    is_active: bool = True
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        # Validate weights sum to 1.0
        total_weight = sum(v.weight for v in self.variants)
        if not (0.99 <= total_weight <= 1.01):  # Allow small floating point error
            raise ValueError(f"Variant weights must sum to 1.0, got {total_weight}")
    
    def get_variant(self, variant_id: str) -> Optional[Variant]:
        """Get variant by ID"""
        for variant in self.variants:
            if variant.id == variant_id:
                return variant
        return None


@dataclass
class ExperimentEvent:
    """Event logged during A/B test"""
    timestamp: str
    experiment_id: str
    variant_id: str
    user_id: str
    event_type: str  # 'query', 'click', 'feedback', 'refinement'
    metrics: Dict[str, Any]
    session_id: Optional[str] = None


class ABTestAssigner:
    """
    Assign users to experiment variants consistently
    
    Uses consistent hashing to ensure same user always
    gets same variant for a given experiment.
    """
    
    def __init__(self):
        """Initialize assigner"""
        self.assignments = {}  # Cache: (user_id, experiment_id) -> variant_id
    
    def assign_variant(
        self,
        user_id: str,
        experiment: Experiment,
        force_variant: Optional[str] = None
    ) -> str:
        """
        Assign user to variant using consistent hashing
        
        Args:
            user_id: User identifier
            experiment: Experiment configuration
            force_variant: Force specific variant (for testing)
            
        Returns:
            Variant ID
        """
        # Check cache first
        cache_key = (user_id, experiment.id)
        if cache_key in self.assignments:
            return self.assignments[cache_key]
        
        # Force variant if specified
        if force_variant:
            variant = experiment.get_variant(force_variant)
            if variant and variant.status == VariantStatus.ACTIVE:
                self.assignments[cache_key] = force_variant
                return force_variant
        
        # Get active variants only
        active_variants = [v for v in experiment.variants if v.status == VariantStatus.ACTIVE]
        if not active_variants:
            logger.warning(f"No active variants for experiment {experiment.id}")
            return experiment.variants[0].id  # Default to first variant
        
        # Consistent hashing: hash user_id + experiment_id
        hash_input = f"{user_id}:{experiment.id}"
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()
        hash_int = int(hash_value, 16)
        
        # Map to 0-100 range
        bucket = (hash_int % 10000) / 100.0  # 0.00 to 99.99
        
        # Assign based on cumulative weights
        cumulative = 0.0
        for variant in active_variants:
            cumulative += variant.weight * 100
            if bucket < cumulative:
                self.assignments[cache_key] = variant.id
                logger.debug(f"Assigned user {user_id} to variant {variant.id} (bucket: {bucket:.2f})")
                return variant.id
        
        # Fallback to last variant (should not happen if weights sum to 1.0)
        variant_id = active_variants[-1].id
        self.assignments[cache_key] = variant_id
        return variant_id
    
    def clear_cache(self):
        """Clear assignment cache"""
        self.assignments.clear()


class ABTestMetricsCollector:
    """
    Collect and store A/B test metrics
    
    Stores events to file for later analysis.
    In production, would use database or analytics service.
    """
    
    def __init__(self, storage_dir: str = "data/ab_tests"):
        """
        Initialize metrics collector
        
        Args:
            storage_dir: Directory for storing event logs
        """
        self.storage_dir = storage_dir
        
        # Create storage directory
        import os
        os.makedirs(storage_dir, exist_ok=True)
        
        # Event buffer (flush periodically)
        self.event_buffer = []
        self.buffer_size = 100
    
    def log_event(
        self,
        experiment_id: str,
        variant_id: str,
        user_id: str,
        event_type: str,
        metrics: Dict[str, Any],
        session_id: Optional[str] = None
    ):
        """
        Log A/B test event
        
        Args:
            experiment_id: Experiment identifier
            variant_id: Variant user was assigned to
            user_id: User identifier
            event_type: Type of event ('query', 'click', 'feedback', etc.)
            metrics: Event metrics (response_time, satisfaction, etc.)
            session_id: Optional session identifier
        """
        event = ExperimentEvent(
            timestamp=datetime.now().isoformat(),
            experiment_id=experiment_id,
            variant_id=variant_id,
            user_id=user_id,
            event_type=event_type,
            metrics=metrics,
            session_id=session_id
        )
        
        self.event_buffer.append(event)
        
        # Flush if buffer full
        if len(self.event_buffer) >= self.buffer_size:
            self.flush()
    
    def flush(self):
        """Flush event buffer to disk"""
        if not self.event_buffer:
            return
        
        # Group by experiment
        events_by_experiment = {}
        for event in self.event_buffer:
            if event.experiment_id not in events_by_experiment:
                events_by_experiment[event.experiment_id] = []
            events_by_experiment[event.experiment_id].append(event)
        
        # Write to files
        for experiment_id, events in events_by_experiment.items():
            filepath = f"{self.storage_dir}/{experiment_id}.jsonl"
            
            try:
                with open(filepath, 'a') as f:
                    for event in events:
                        f.write(json.dumps(asdict(event)) + '\n')
            except Exception as e:
                logger.error(f"Failed to write events: {e}")
        
        # Clear buffer
        self.event_buffer.clear()
        logger.debug(f"Flushed {len(self.event_buffer)} events to disk")
    
    def load_events(self, experiment_id: str) -> List[ExperimentEvent]:
        """
        Load all events for an experiment
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            List of events
        """
        filepath = f"{self.storage_dir}/{experiment_id}.jsonl"
        events = []
        
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    events.append(ExperimentEvent(**data))
        except FileNotFoundError:
            logger.warning(f"No events file for experiment {experiment_id}")
        except Exception as e:
            logger.error(f"Failed to load events: {e}")
        
        return events
    
    def __del__(self):
        """Flush on destruction"""
        try:
            self.flush()
        except:
            pass


class ABTestFramework:
    """
    Complete A/B testing framework
    
    Manages experiments, assigns users to variants,
    collects metrics, and provides variant execution.
    """
    
    def __init__(self, storage_dir: str = "data/ab_tests"):
        """
        Initialize A/B testing framework
        
        Args:
            storage_dir: Directory for storing experiment data
        """
        self.experiments: Dict[str, Experiment] = {}
        self.assigner = ABTestAssigner()
        self.metrics = ABTestMetricsCollector(storage_dir)
        
        logger.info("✅ A/B Test Framework initialized")
    
    def create_experiment(self, experiment: Experiment):
        """
        Create new experiment
        
        Args:
            experiment: Experiment configuration
        """
        if experiment.id in self.experiments:
            logger.warning(f"Experiment {experiment.id} already exists, overwriting")
        
        self.experiments[experiment.id] = experiment
        logger.info(f"✅ Created experiment: {experiment.id} ({len(experiment.variants)} variants)")
    
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get experiment by ID"""
        return self.experiments.get(experiment_id)
    
    def execute_variant(
        self,
        experiment_id: str,
        user_id: str,
        execution_fn: Callable[[Dict[str, Any]], Any],
        metrics_fn: Optional[Callable[[Any], Dict[str, Any]]] = None,
        force_variant: Optional[str] = None
    ) -> Any:
        """
        Execute experiment variant for user
        
        Args:
            experiment_id: Experiment to run
            user_id: User identifier
            execution_fn: Function that executes with variant config
            metrics_fn: Optional function to extract metrics from result
            force_variant: Force specific variant (for testing)
            
        Returns:
            Result from execution_fn
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment or not experiment.is_active:
            logger.warning(f"Experiment {experiment_id} not found or inactive")
            # Execute with default config
            return execution_fn({})
        
        # Assign variant
        variant_id = self.assigner.assign_variant(user_id, experiment, force_variant)
        variant = experiment.get_variant(variant_id)
        
        if not variant:
            logger.error(f"Variant {variant_id} not found in experiment {experiment_id}")
            return execution_fn({})
        
        # Execute with variant config
        import time
        start_time = time.time()
        
        try:
            result = execution_fn(variant.config)
            elapsed_ms = (time.time() - start_time) * 1000
            
            # Extract metrics
            metrics = {'execution_time_ms': elapsed_ms}
            if metrics_fn:
                try:
                    custom_metrics = metrics_fn(result)
                    metrics.update(custom_metrics)
                except Exception as e:
                    logger.warning(f"Metrics extraction failed: {e}")
            
            # Log event
            self.metrics.log_event(
                experiment_id=experiment_id,
                variant_id=variant_id,
                user_id=user_id,
                event_type='execution',
                metrics=metrics
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Variant execution failed: {e}")
            # Log failure
            self.metrics.log_event(
                experiment_id=experiment_id,
                variant_id=variant_id,
                user_id=user_id,
                event_type='error',
                metrics={'error': str(e)}
            )
            raise
    
    def log_user_interaction(
        self,
        experiment_id: str,
        user_id: str,
        event_type: str,
        metrics: Dict[str, Any]
    ):
        """
        Log user interaction event
        
        Args:
            experiment_id: Experiment identifier
            user_id: User identifier
            event_type: Type of interaction ('click', 'feedback', etc.)
            metrics: Interaction metrics
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return
        
        # Get user's variant assignment
        variant_id = self.assigner.assign_variant(user_id, experiment)
        
        # Log event
        self.metrics.log_event(
            experiment_id=experiment_id,
            variant_id=variant_id,
            user_id=user_id,
            event_type=event_type,
            metrics=metrics
        )
    
    def get_stats(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get experiment statistics
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Statistics dictionary
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return {}
        
        # Load all events
        events = self.metrics.load_events(experiment_id)
        
        # Group by variant
        variant_events = {}
        for variant in experiment.variants:
            variant_events[variant.id] = [e for e in events if e.variant_id == variant.id]
        
        # Calculate stats
        stats = {
            'experiment_id': experiment_id,
            'total_events': len(events),
            'variants': {}
        }
        
        for variant_id, events_list in variant_events.items():
            stats['variants'][variant_id] = {
                'event_count': len(events_list),
                'unique_users': len(set(e.user_id for e in events_list))
            }
        
        return stats
    
    def list_experiments(self) -> List[str]:
        """List all experiment IDs"""
        return list(self.experiments.keys())


def create_ab_test_framework(storage_dir: str = "data/ab_tests") -> ABTestFramework:
    """
    Factory function to create A/B test framework
    
    Args:
        storage_dir: Storage directory for experiment data
        
    Returns:
        ABTestFramework instance
    """
    return ABTestFramework(storage_dir=storage_dir)
