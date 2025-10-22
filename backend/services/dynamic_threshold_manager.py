"""
Dynamic Threshold Manager
Adjusts confidence thresholds based on context and query characteristics
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

from services.context_aware_classifier import ContextFeatures

logger = logging.getLogger(__name__)


# Intent-specific threshold configuration
THRESHOLD_CONFIG = {
    "restaurant_query": {
        "base": 0.70,
        "with_context": 0.60,  # Lower threshold for follow-ups
        "with_entities": 0.65,
        "with_location": 0.63
    },
    "restaurant_recommendation": {
        "base": 0.70,
        "with_context": 0.60,
        "with_entities": 0.65,
        "with_location": 0.63
    },
    "attraction_query": {
        "base": 0.70,
        "with_context": 0.60,
        "with_entities": 0.65,
        "with_location": 0.62
    },
    "museum_query": {
        "base": 0.70,
        "with_context": 0.60,
        "with_entities": 0.65,
        "with_location": 0.62
    },
    "transport_query": {
        "base": 0.75,  # Higher threshold (more specific)
        "with_context": 0.65,
        "with_entities": 0.70,
        "with_location": 0.68
    },
    "event_query": {
        "base": 0.72,
        "with_context": 0.62,
        "with_entities": 0.67,
        "with_location": 0.64
    },
    "general_info": {
        "base": 0.50,  # Lower threshold (catch-all)
        "with_context": 0.40,
        "with_entities": 0.45,
        "with_location": 0.43
    },
    "greeting": {
        "base": 0.80,  # High threshold (very specific)
        "with_context": 0.75,
        "with_entities": 0.78,
        "with_location": 0.78
    }
}


@dataclass
class ThresholdDecision:
    """Represents a threshold decision with reasoning"""
    threshold: float
    base_threshold: float
    adjustments: Dict[str, float]
    reasoning: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'threshold': self.threshold,
            'base_threshold': self.base_threshold,
            'adjustments': self.adjustments,
            'total_adjustment': self.threshold - self.base_threshold,
            'reasoning': self.reasoning
        }


class DynamicThresholdManager:
    """
    Manages adaptive confidence thresholds based on context and query characteristics
    """
    
    def __init__(self):
        """Initialize threshold manager"""
        self.threshold_config = THRESHOLD_CONFIG
        self.default_base_threshold = 0.70
        
        logger.info("✅ DynamicThresholdManager initialized")
    
    def get_adaptive_threshold(
        self,
        intent: str,
        context_features: Optional[ContextFeatures] = None,
        entities: Optional[Dict[str, Any]] = None,
        query_length: Optional[int] = None
    ) -> ThresholdDecision:
        """
        Calculate adaptive confidence threshold for an intent
        
        Args:
            intent: Intent to get threshold for
            context_features: Context features (if available)
            entities: Extracted entities (if available)
            query_length: Query length in words (if available)
            
        Returns:
            ThresholdDecision with threshold and reasoning
        """
        # Get base configuration for intent
        config = self.threshold_config.get(intent, {})
        base_threshold = config.get("base", self.default_base_threshold)
        
        threshold = base_threshold
        adjustments = {}
        reasoning_parts = [f"Base threshold: {base_threshold:.2f}"]
        
        # Adjustment 1: Context-based
        if context_features and context_features.has_previous_intent:
            context_threshold = config.get("with_context", base_threshold - 0.10)
            adjustment = context_threshold - base_threshold
            if adjustment != 0:
                adjustments['context'] = adjustment
                threshold = context_threshold
                reasoning_parts.append(
                    f"Context adjustment: {adjustment:+.2f} (previous intent: {context_features.previous_intent})"
                )
        
        # Adjustment 2: Entity-based
        if entities and len(entities) >= 2:
            entity_threshold = config.get("with_entities", base_threshold - 0.05)
            adjustment = entity_threshold - base_threshold
            if adjustment != 0 and 'context' not in adjustments:  # Don't double-adjust
                adjustments['entities'] = adjustment
                threshold = entity_threshold
                reasoning_parts.append(
                    f"Entity adjustment: {adjustment:+.2f} ({len(entities)} entities)"
                )
        
        # Adjustment 3: Location context
        if entities and 'locations' in entities and entities['locations']:
            location_threshold = config.get("with_location", base_threshold - 0.07)
            if location_threshold < threshold:  # Only apply if it lowers threshold
                adjustment = location_threshold - base_threshold
                adjustments['location'] = adjustment
                threshold = location_threshold
                reasoning_parts.append(
                    f"Location adjustment: {adjustment:+.2f} (location specified)"
                )
        
        # Adjustment 4: Follow-up query
        if context_features and context_features.follow_up_likelihood > 0.7:
            follow_up_adjustment = -0.08 * context_features.follow_up_likelihood
            adjustments['follow_up'] = follow_up_adjustment
            threshold += follow_up_adjustment
            reasoning_parts.append(
                f"Follow-up adjustment: {follow_up_adjustment:+.2f} (likelihood: {context_features.follow_up_likelihood:.2f})"
            )
        
        # Adjustment 5: Conversation depth (more confident with longer conversations)
        if context_features and context_features.conversation_depth >= 3:
            depth_adjustment = -0.03 * min(context_features.conversation_depth / 5, 1.0)
            adjustments['conversation_depth'] = depth_adjustment
            threshold += depth_adjustment
            reasoning_parts.append(
                f"Depth adjustment: {depth_adjustment:+.2f} ({context_features.conversation_depth} turns)"
            )
        
        # Adjustment 6: Query length (short queries in context often valid)
        if query_length and query_length <= 2 and context_features and context_features.conversation_depth >= 2:
            short_query_adjustment = -0.05
            adjustments['short_query'] = short_query_adjustment
            threshold += short_query_adjustment
            reasoning_parts.append(
                f"Short query adjustment: {short_query_adjustment:+.2f} ({query_length} words)"
            )
        
        # Adjustment 7: Recent conversation (within last 30 seconds)
        if context_features and context_features.time_since_last_turn < 30:
            recency_adjustment = -0.04
            adjustments['recency'] = recency_adjustment
            threshold += recency_adjustment
            reasoning_parts.append(
                f"Recency adjustment: {recency_adjustment:+.2f} ({context_features.time_since_last_turn:.0f}s gap)"
            )
        
        # Ensure threshold stays within reasonable bounds
        threshold = max(0.30, min(threshold, 0.95))
        
        reasoning = " | ".join(reasoning_parts)
        
        logger.debug(f"🎯 Adaptive threshold for {intent}: {threshold:.2f}")
        for key, value in adjustments.items():
            logger.debug(f"  • {key}: {value:+.3f}")
        
        return ThresholdDecision(
            threshold=threshold,
            base_threshold=base_threshold,
            adjustments=adjustments,
            reasoning=reasoning
        )
    
    def should_accept(
        self,
        intent: str,
        confidence: float,
        context_features: Optional[ContextFeatures] = None,
        entities: Optional[Dict[str, Any]] = None,
        query_length: Optional[int] = None
    ) -> tuple[bool, ThresholdDecision]:
        """
        Determine if classification should be accepted based on adaptive threshold
        
        Args:
            intent: Classified intent
            confidence: Classification confidence
            context_features: Context features
            entities: Extracted entities
            query_length: Query length in words
            
        Returns:
            Tuple of (should_accept, threshold_decision)
        """
        threshold_decision = self.get_adaptive_threshold(
            intent, context_features, entities, query_length
        )
        
        should_accept = confidence >= threshold_decision.threshold
        
        if should_accept:
            margin = confidence - threshold_decision.threshold
            logger.info(
                f"✅ Classification accepted: {intent} "
                f"(confidence: {confidence:.3f}, threshold: {threshold_decision.threshold:.3f}, margin: +{margin:.3f})"
            )
        else:
            deficit = threshold_decision.threshold - confidence
            logger.warning(
                f"❌ Classification rejected: {intent} "
                f"(confidence: {confidence:.3f}, threshold: {threshold_decision.threshold:.3f}, deficit: -{deficit:.3f})"
            )
        
        return should_accept, threshold_decision
    
    def get_threshold_for_intent(self, intent: str, threshold_type: str = "base") -> float:
        """
        Get a specific threshold value for an intent
        
        Args:
            intent: Intent name
            threshold_type: Type of threshold (base, with_context, with_entities, with_location)
            
        Returns:
            Threshold value
        """
        config = self.threshold_config.get(intent, {})
        return config.get(threshold_type, self.default_base_threshold)
    
    def update_threshold_config(self, intent: str, threshold_type: str, value: float):
        """
        Update threshold configuration (for testing or tuning)
        
        Args:
            intent: Intent to update
            threshold_type: Type of threshold
            value: New threshold value
        """
        if intent not in self.threshold_config:
            self.threshold_config[intent] = {}
        
        self.threshold_config[intent][threshold_type] = value
        logger.info(f"📊 Updated threshold: {intent}.{threshold_type} = {value:.2f}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get threshold manager statistics"""
        intents_configured = len(self.threshold_config)
        
        # Calculate average thresholds
        avg_base = sum(
            config.get("base", self.default_base_threshold)
            for config in self.threshold_config.values()
        ) / intents_configured if intents_configured > 0 else self.default_base_threshold
        
        avg_context = sum(
            config.get("with_context", config.get("base", self.default_base_threshold) - 0.10)
            for config in self.threshold_config.values()
        ) / intents_configured if intents_configured > 0 else self.default_base_threshold - 0.10
        
        return {
            'intents_configured': intents_configured,
            'default_base_threshold': self.default_base_threshold,
            'avg_base_threshold': avg_base,
            'avg_context_threshold': avg_context,
            'threshold_reduction_avg': avg_base - avg_context
        }


# Singleton instance
_threshold_manager_instance = None

def get_threshold_manager() -> DynamicThresholdManager:
    """Get or create threshold manager singleton"""
    global _threshold_manager_instance
    if _threshold_manager_instance is None:
        _threshold_manager_instance = DynamicThresholdManager()
    return _threshold_manager_instance
