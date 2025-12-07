"""
continuous_learning.py - Continuous Learning Pipeline

Automated learning system that:
- Collects user feedback continuously
- Learns patterns from successful/failed queries
- Updates models incrementally
- Deploys improvements via canary releases
- Monitors performance in production

Author: AI Istanbul Team
Date: December 7, 2025
"""

import logging
import time
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of user feedback."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    CORRECTION = "correction"
    RATING = "rating"


class ModelVersion(Enum):
    """Model deployment stages."""
    TRAINING = "training"
    CANARY = "canary"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"


@dataclass
class FeedbackEvent:
    """User feedback event."""
    id: str
    user_id: str
    query: str
    response: str
    feedback_type: str
    rating: Optional[float]
    correction: Optional[str]
    metadata: Dict[str, Any]
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class LearningPattern:
    """Learned pattern from feedback."""
    pattern_id: str
    pattern_type: str  # "intent", "entity", "context", "response"
    pattern: Dict[str, Any]
    confidence: float
    support: int  # Number of examples supporting this pattern
    created_at: float
    last_updated: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ModelUpdate:
    """Model update information."""
    version: str
    model_type: str
    changes: List[str]
    metrics: Dict[str, float]
    status: str
    deployed_at: Optional[float]
    created_at: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class FeedbackCollector:
    """
    Collects and processes user feedback for continuous learning.
    
    Features:
    - Real-time feedback collection
    - Feedback aggregation and analysis
    - Pattern extraction
    - Quality assessment
    """
    
    def __init__(self, buffer_size: int = 1000):
        """
        Initialize feedback collector.
        
        Args:
            buffer_size: Maximum feedback events to keep in memory
        """
        self.buffer_size = buffer_size
        self.feedback_buffer = deque(maxlen=buffer_size)
        self.feedback_stats = defaultdict(lambda: {"count": 0, "avg_rating": 0.0})
        
        logger.info(f"✅ FeedbackCollector initialized (buffer_size={buffer_size})")
    
    def collect_feedback(
        self,
        user_id: str,
        query: str,
        response: str,
        feedback_type: str,
        rating: Optional[float] = None,
        correction: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FeedbackEvent:
        """
        Collect a feedback event.
        
        Args:
            user_id: User ID
            query: Original query
            response: System response
            feedback_type: Type of feedback
            rating: Rating (0-5)
            correction: User correction
            metadata: Additional metadata
            
        Returns:
            FeedbackEvent object
        """
        event = FeedbackEvent(
            id=f"fb_{int(time.time() * 1000)}_{user_id[:8]}",
            user_id=user_id,
            query=query,
            response=response,
            feedback_type=feedback_type,
            rating=rating,
            correction=correction,
            metadata=metadata or {},
            timestamp=time.time()
        )
        
        # Add to buffer
        self.feedback_buffer.append(event)
        
        # Update stats
        self._update_stats(event)
        
        logger.info(f"📝 Collected feedback: {feedback_type} (rating={rating})")
        return event
    
    def _update_stats(self, event: FeedbackEvent):
        """Update feedback statistics."""
        key = event.feedback_type
        stats = self.feedback_stats[key]
        
        stats["count"] += 1
        
        if event.rating is not None:
            # Update running average
            n = stats["count"]
            old_avg = stats["avg_rating"]
            stats["avg_rating"] = old_avg + (event.rating - old_avg) / n
    
    def get_recent_feedback(
        self,
        limit: int = 100,
        feedback_type: Optional[str] = None,
        min_rating: Optional[float] = None
    ) -> List[FeedbackEvent]:
        """
        Get recent feedback events.
        
        Args:
            limit: Maximum events to return
            feedback_type: Filter by feedback type
            min_rating: Minimum rating threshold
            
        Returns:
            List of FeedbackEvent objects
        """
        events = list(self.feedback_buffer)
        
        # Apply filters
        if feedback_type:
            events = [e for e in events if e.feedback_type == feedback_type]
        
        if min_rating is not None:
            events = [e for e in events if e.rating and e.rating >= min_rating]
        
        # Sort by timestamp (most recent first)
        events.sort(key=lambda e: e.timestamp, reverse=True)
        
        return events[:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get feedback statistics."""
        total_count = sum(stats["count"] for stats in self.feedback_stats.values())
        
        return {
            "total_feedback": total_count,
            "by_type": dict(self.feedback_stats),
            "buffer_size": len(self.feedback_buffer),
            "buffer_capacity": self.buffer_size
        }


class PatternLearner:
    """
    Learns patterns from feedback data.
    
    Features:
    - Intent pattern extraction
    - Entity pattern recognition
    - Response quality patterns
    - Contextual patterns
    """
    
    def __init__(self, min_support: int = 3, min_confidence: float = 0.7):
        """
        Initialize pattern learner.
        
        Args:
            min_support: Minimum examples to establish pattern
            min_confidence: Minimum confidence threshold
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.learned_patterns = {}
        
        logger.info(f"✅ PatternLearner initialized (support={min_support}, confidence={min_confidence})")
    
    def learn_from_feedback(
        self,
        feedback_events: List[FeedbackEvent]
    ) -> List[LearningPattern]:
        """
        Learn patterns from feedback events.
        
        Args:
            feedback_events: List of feedback events
            
        Returns:
            List of learned patterns
        """
        patterns = []
        
        # Learn intent patterns
        intent_patterns = self._learn_intent_patterns(feedback_events)
        patterns.extend(intent_patterns)
        
        # Learn entity patterns
        entity_patterns = self._learn_entity_patterns(feedback_events)
        patterns.extend(entity_patterns)
        
        # Learn response patterns
        response_patterns = self._learn_response_patterns(feedback_events)
        patterns.extend(response_patterns)
        
        logger.info(f"🧠 Learned {len(patterns)} patterns from {len(feedback_events)} feedback events")
        return patterns
    
    def _learn_intent_patterns(self, events: List[FeedbackEvent]) -> List[LearningPattern]:
        """Learn intent-related patterns."""
        patterns = []
        
        # Group by query similarity
        query_groups = self._group_similar_queries(events)
        
        for group_id, group_events in query_groups.items():
            if len(group_events) < self.min_support:
                continue
            
            # Calculate success rate
            positive_count = sum(1 for e in group_events if e.feedback_type == FeedbackType.POSITIVE.value)
            confidence = positive_count / len(group_events)
            
            if confidence >= self.min_confidence:
                # Extract common patterns
                pattern = self._extract_query_pattern(group_events)
                
                learning_pattern = LearningPattern(
                    pattern_id=f"intent_{group_id}",
                    pattern_type="intent",
                    pattern=pattern,
                    confidence=confidence,
                    support=len(group_events),
                    created_at=time.time(),
                    last_updated=time.time()
                )
                
                patterns.append(learning_pattern)
                self.learned_patterns[learning_pattern.pattern_id] = learning_pattern
        
        return patterns
    
    def _learn_entity_patterns(self, events: List[FeedbackEvent]) -> List[LearningPattern]:
        """Learn entity extraction patterns."""
        patterns = []
        
        # Extract entities from corrections
        correction_events = [e for e in events if e.correction]
        
        if len(correction_events) < self.min_support:
            return patterns
        
        # Analyze corrections for entity patterns
        entity_corrections = defaultdict(list)
        
        for event in correction_events:
            # Simple entity extraction (can be enhanced)
            entities = self._extract_entities(event.query, event.correction)
            for entity_type, entity_value in entities:
                entity_corrections[entity_type].append({
                    "original": event.query,
                    "correction": event.correction,
                    "entity": entity_value
                })
        
        # Create patterns for frequent corrections
        for entity_type, corrections in entity_corrections.items():
            if len(corrections) >= self.min_support:
                pattern = LearningPattern(
                    pattern_id=f"entity_{entity_type}_{int(time.time())}",
                    pattern_type="entity",
                    pattern={
                        "entity_type": entity_type,
                        "examples": corrections[:10],  # Keep top 10 examples
                        "frequency": len(corrections)
                    },
                    confidence=len(corrections) / len(correction_events),
                    support=len(corrections),
                    created_at=time.time(),
                    last_updated=time.time()
                )
                
                patterns.append(pattern)
                self.learned_patterns[pattern.pattern_id] = pattern
        
        return patterns
    
    def _learn_response_patterns(self, events: List[FeedbackEvent]) -> List[LearningPattern]:
        """Learn response quality patterns."""
        patterns = []
        
        # Analyze high-rated vs low-rated responses
        high_rated = [e for e in events if e.rating and e.rating >= 4.0]
        low_rated = [e for e in events if e.rating and e.rating <= 2.0]
        
        if len(high_rated) < self.min_support or len(low_rated) < self.min_support:
            return patterns
        
        # Identify characteristics of good responses
        good_response_pattern = self._analyze_response_characteristics(high_rated)
        bad_response_pattern = self._analyze_response_characteristics(low_rated)
        
        if good_response_pattern:
            pattern = LearningPattern(
                pattern_id=f"response_quality_good_{int(time.time())}",
                pattern_type="response",
                pattern={
                    "quality": "good",
                    "characteristics": good_response_pattern
                },
                confidence=len(high_rated) / len(events),
                support=len(high_rated),
                created_at=time.time(),
                last_updated=time.time()
            )
            patterns.append(pattern)
            self.learned_patterns[pattern.pattern_id] = pattern
        
        return patterns
    
    def _group_similar_queries(self, events: List[FeedbackEvent]) -> Dict[str, List[FeedbackEvent]]:
        """Group similar queries together."""
        # Simple grouping by first word (can be enhanced with embeddings)
        groups = defaultdict(list)
        
        for event in events:
            first_word = event.query.strip().split()[0].lower() if event.query.strip() else "unknown"
            groups[first_word].append(event)
        
        return groups
    
    def _extract_query_pattern(self, events: List[FeedbackEvent]) -> Dict[str, Any]:
        """Extract common pattern from similar queries."""
        # Simple pattern extraction (can be enhanced)
        return {
            "example_queries": [e.query for e in events[:5]],
            "count": len(events),
            "avg_rating": np.mean([e.rating for e in events if e.rating]) if any(e.rating for e in events) else None
        }
    
    def _extract_entities(self, query: str, correction: str) -> List[Tuple[str, str]]:
        """Extract entities from query and correction."""
        # Placeholder - implement actual entity extraction
        return []
    
    def _analyze_response_characteristics(self, events: List[FeedbackEvent]) -> Dict[str, Any]:
        """Analyze characteristics of responses."""
        if not events:
            return {}
        
        avg_length = np.mean([len(e.response) for e in events])
        avg_rating = np.mean([e.rating for e in events if e.rating])
        
        return {
            "avg_length": avg_length,
            "avg_rating": avg_rating,
            "sample_count": len(events)
        }
    
    def get_patterns(
        self,
        pattern_type: Optional[str] = None,
        min_confidence: Optional[float] = None
    ) -> List[LearningPattern]:
        """
        Get learned patterns.
        
        Args:
            pattern_type: Filter by pattern type
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of patterns
        """
        patterns = list(self.learned_patterns.values())
        
        if pattern_type:
            patterns = [p for p in patterns if p.pattern_type == pattern_type]
        
        if min_confidence:
            patterns = [p for p in patterns if p.confidence >= min_confidence]
        
        return patterns


class CanaryDeployment:
    """
    Manages canary deployments for model updates.
    
    Features:
    - Gradual traffic shifting
    - Performance monitoring
    - Automatic rollback on failures
    - A/B testing integration
    """
    
    def __init__(self, initial_traffic: float = 0.1, max_traffic: float = 1.0):
        """
        Initialize canary deployment manager.
        
        Args:
            initial_traffic: Initial traffic percentage to canary
            max_traffic: Maximum traffic percentage
        """
        self.initial_traffic = initial_traffic
        self.max_traffic = max_traffic
        self.deployments = {}
        
        logger.info(f"✅ CanaryDeployment initialized (initial={initial_traffic}, max={max_traffic})")
    
    def deploy_canary(
        self,
        model_version: str,
        model_type: str,
        traffic_percentage: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Deploy a new model version as canary.
        
        Args:
            model_version: Model version identifier
            model_type: Type of model
            traffic_percentage: Initial traffic percentage (optional)
            
        Returns:
            Deployment info
        """
        traffic = traffic_percentage or self.initial_traffic
        
        deployment = {
            "model_version": model_version,
            "model_type": model_type,
            "status": ModelVersion.CANARY.value,
            "traffic_percentage": traffic,
            "deployed_at": time.time(),
            "metrics": {
                "requests": 0,
                "successes": 0,
                "failures": 0,
                "avg_latency": 0.0,
                "error_rate": 0.0
            }
        }
        
        deployment_id = f"{model_type}_{model_version}"
        self.deployments[deployment_id] = deployment
        
        logger.info(f"🚀 Canary deployed: {deployment_id} ({traffic*100:.1f}% traffic)")
        return deployment
    
    def update_traffic(
        self,
        deployment_id: str,
        traffic_percentage: float
    ) -> Dict[str, Any]:
        """
        Update traffic percentage for a canary deployment.
        
        Args:
            deployment_id: Deployment identifier
            traffic_percentage: New traffic percentage
            
        Returns:
            Updated deployment info
        """
        if deployment_id not in self.deployments:
            raise ValueError(f"Deployment not found: {deployment_id}")
        
        deployment = self.deployments[deployment_id]
        old_traffic = deployment["traffic_percentage"]
        deployment["traffic_percentage"] = min(traffic_percentage, self.max_traffic)
        
        logger.info(f"📊 Traffic updated: {deployment_id} ({old_traffic*100:.1f}% → {traffic_percentage*100:.1f}%)")
        return deployment
    
    def record_request(
        self,
        deployment_id: str,
        success: bool,
        latency: float
    ):
        """
        Record a request to the canary deployment.
        
        Args:
            deployment_id: Deployment identifier
            success: Whether request succeeded
            latency: Request latency in seconds
        """
        if deployment_id not in self.deployments:
            return
        
        deployment = self.deployments[deployment_id]
        metrics = deployment["metrics"]
        
        # Update metrics
        metrics["requests"] += 1
        if success:
            metrics["successes"] += 1
        else:
            metrics["failures"] += 1
        
        # Update average latency (running average)
        n = metrics["requests"]
        old_avg = metrics["avg_latency"]
        metrics["avg_latency"] = old_avg + (latency - old_avg) / n
        
        # Update error rate
        metrics["error_rate"] = metrics["failures"] / metrics["requests"]
    
    def should_promote(
        self,
        deployment_id: str,
        min_requests: int = 100,
        max_error_rate: float = 0.05,
        max_latency: float = 2.0
    ) -> Tuple[bool, str]:
        """
        Check if canary should be promoted to production.
        
        Args:
            deployment_id: Deployment identifier
            min_requests: Minimum requests before promotion
            max_error_rate: Maximum acceptable error rate
            max_latency: Maximum acceptable latency
            
        Returns:
            (should_promote, reason)
        """
        if deployment_id not in self.deployments:
            return False, "Deployment not found"
        
        deployment = self.deployments[deployment_id]
        metrics = deployment["metrics"]
        
        # Check minimum requests
        if metrics["requests"] < min_requests:
            return False, f"Insufficient requests ({metrics['requests']}/{min_requests})"
        
        # Check error rate
        if metrics["error_rate"] > max_error_rate:
            return False, f"High error rate ({metrics['error_rate']*100:.2f}% > {max_error_rate*100:.2f}%)"
        
        # Check latency
        if metrics["avg_latency"] > max_latency:
            return False, f"High latency ({metrics['avg_latency']:.2f}s > {max_latency:.2f}s)"
        
        return True, "All criteria met"
    
    def promote_to_production(self, deployment_id: str) -> Dict[str, Any]:
        """
        Promote canary to production.
        
        Args:
            deployment_id: Deployment identifier
            
        Returns:
            Updated deployment info
        """
        if deployment_id not in self.deployments:
            raise ValueError(f"Deployment not found: {deployment_id}")
        
        deployment = self.deployments[deployment_id]
        deployment["status"] = ModelVersion.PRODUCTION.value
        deployment["traffic_percentage"] = 1.0
        deployment["promoted_at"] = time.time()
        
        logger.info(f"✅ Promoted to production: {deployment_id}")
        return deployment
    
    def rollback(self, deployment_id: str) -> Dict[str, Any]:
        """
        Rollback a canary deployment.
        
        Args:
            deployment_id: Deployment identifier
            
        Returns:
            Updated deployment info
        """
        if deployment_id not in self.deployments:
            raise ValueError(f"Deployment not found: {deployment_id}")
        
        deployment = self.deployments[deployment_id]
        deployment["status"] = ModelVersion.DEPRECATED.value
        deployment["traffic_percentage"] = 0.0
        deployment["rolled_back_at"] = time.time()
        
        logger.warning(f"⚠️ Rolled back: {deployment_id}")
        return deployment
    
    def get_deployment(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get deployment info."""
        return self.deployments.get(deployment_id)
    
    def list_deployments(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all deployments."""
        deployments = list(self.deployments.values())
        
        if status:
            deployments = [d for d in deployments if d["status"] == status]
        
        return deployments


class ContinuousLearningPipeline:
    """
    Main continuous learning pipeline orchestrator.
    
    Integrates:
    - Feedback collection
    - Pattern learning
    - Model updates
    - Canary deployments
    - Performance monitoring
    """
    
    def __init__(
        self,
        feedback_collector: Optional[FeedbackCollector] = None,
        pattern_learner: Optional[PatternLearner] = None,
        canary_deployment: Optional[CanaryDeployment] = None,
        learning_interval: int = 3600,  # 1 hour
        deployment_interval: int = 86400  # 24 hours
    ):
        """
        Initialize continuous learning pipeline.
        
        Args:
            feedback_collector: Feedback collector instance
            pattern_learner: Pattern learner instance
            canary_deployment: Canary deployment manager
            learning_interval: Seconds between learning cycles
            deployment_interval: Seconds between deployments
        """
        self.feedback_collector = feedback_collector or FeedbackCollector()
        self.pattern_learner = pattern_learner or PatternLearner()
        self.canary_deployment = canary_deployment or CanaryDeployment()
        
        self.learning_interval = learning_interval
        self.deployment_interval = deployment_interval
        
        self.last_learning_run = 0
        self.last_deployment = 0
        self.model_versions = []
        
        logger.info("✅ ContinuousLearningPipeline initialized")
    
    async def run_learning_cycle(self) -> Dict[str, Any]:
        """
        Run a complete learning cycle.
        
        Returns:
            Learning cycle results
        """
        start_time = time.time()
        
        # Step 1: Get recent feedback
        feedback_events = self.feedback_collector.get_recent_feedback(limit=1000)
        
        if len(feedback_events) < 10:
            logger.info("⏭️ Insufficient feedback for learning cycle")
            return {
                "status": "skipped",
                "reason": "insufficient_feedback",
                "feedback_count": len(feedback_events)
            }
        
        # Step 2: Learn patterns
        patterns = self.pattern_learner.learn_from_feedback(feedback_events)
        
        # Step 3: Evaluate if model update is needed
        should_update = len(patterns) >= 3  # At least 3 new patterns
        
        if not should_update:
            logger.info("⏭️ No significant patterns learned")
            return {
                "status": "no_update_needed",
                "feedback_count": len(feedback_events),
                "patterns_learned": len(patterns)
            }
        
        # Step 4: Create model update
        model_version = f"v{int(time.time())}"
        model_update = ModelUpdate(
            version=model_version,
            model_type="intent_classifier",
            changes=[p.pattern_id for p in patterns],
            metrics={
                "patterns_learned": len(patterns),
                "feedback_processed": len(feedback_events)
            },
            status="ready",
            deployed_at=None,
            created_at=time.time()
        )
        
        self.model_versions.append(model_update)
        
        # Step 5: Trigger canary deployment (if interval met)
        deployment_result = None
        if time.time() - self.last_deployment >= self.deployment_interval:
            deployment_result = self.canary_deployment.deploy_canary(
                model_version=model_version,
                model_type="intent_classifier"
            )
            self.last_deployment = time.time()
        
        duration = time.time() - start_time
        
        logger.info(f"✅ Learning cycle completed in {duration:.2f}s")
        
        return {
            "status": "success",
            "feedback_count": len(feedback_events),
            "patterns_learned": len(patterns),
            "model_version": model_version,
            "deployment": deployment_result,
            "duration": duration
        }
    
    async def monitor_canaries(self) -> Dict[str, Any]:
        """
        Monitor active canary deployments and promote/rollback as needed.
        
        Returns:
            Monitoring results
        """
        canaries = self.canary_deployment.list_deployments(status=ModelVersion.CANARY.value)
        
        results = {
            "evaluated": 0,
            "promoted": 0,
            "rolled_back": 0,
            "unchanged": 0
        }
        
        for canary in canaries:
            deployment_id = f"{canary['model_type']}_{canary['model_version']}"
            results["evaluated"] += 1
            
            # Check if should promote
            should_promote, reason = self.canary_deployment.should_promote(deployment_id)
            
            if should_promote:
                self.canary_deployment.promote_to_production(deployment_id)
                results["promoted"] += 1
                logger.info(f"✅ Promoted canary: {deployment_id}")
            elif "High error rate" in reason or "High latency" in reason:
                self.canary_deployment.rollback(deployment_id)
                results["rolled_back"] += 1
                logger.warning(f"⚠️ Rolled back canary: {deployment_id} - {reason}")
            else:
                results["unchanged"] += 1
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "feedback": self.feedback_collector.get_statistics(),
            "patterns": {
                "total": len(self.pattern_learner.learned_patterns),
                "by_type": self._count_patterns_by_type()
            },
            "deployments": {
                "total": len(self.canary_deployment.deployments),
                "by_status": self._count_deployments_by_status()
            },
            "model_versions": len(self.model_versions),
            "last_learning_run": self.last_learning_run,
            "last_deployment": self.last_deployment
        }
    
    def _count_patterns_by_type(self) -> Dict[str, int]:
        """Count patterns by type."""
        counts = defaultdict(int)
        for pattern in self.pattern_learner.learned_patterns.values():
            counts[pattern.pattern_type] += 1
        return dict(counts)
    
    def _count_deployments_by_status(self) -> Dict[str, int]:
        """Count deployments by status."""
        counts = defaultdict(int)
        for deployment in self.canary_deployment.deployments.values():
            counts[deployment["status"]] += 1
        return dict(counts)


# Global pipeline instance
_pipeline: Optional[ContinuousLearningPipeline] = None


def get_pipeline() -> ContinuousLearningPipeline:
    """Get or create global pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = ContinuousLearningPipeline()
    return _pipeline


if __name__ == "__main__":
    # Demo usage
    pipeline = ContinuousLearningPipeline()
    
    # Simulate feedback collection
    for i in range(20):
        pipeline.feedback_collector.collect_feedback(
            user_id=f"user_{i % 5}",
            query=f"Show me restaurants in Taksim",
            response=f"Here are restaurants in Taksim...",
            feedback_type=FeedbackType.POSITIVE.value if i % 3 == 0 else FeedbackType.RATING.value,
            rating=4.5 if i % 3 == 0 else 3.0
        )
    
    # Run learning cycle
    async def demo():
        result = await pipeline.run_learning_cycle()
        print(f"\nLearning cycle result: {json.dumps(result, indent=2)}")
        
        stats = pipeline.get_statistics()
        print(f"\nPipeline statistics: {json.dumps(stats, indent=2)}")
    
    asyncio.run(demo())
