"""
Context-Aware Intent Classifier
Enhances classification using conversation context and confidence boosting
"""

import logging
import re
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from services.conversation_context_manager import (
    ConversationContext,
    ConversationContextManager,
    get_context_manager
)

logger = logging.getLogger(__name__)


@dataclass
class ContextFeatures:
    """Features extracted from conversation context"""
    has_previous_intent: bool
    previous_intent: Optional[str]
    intent_sequence_pattern: str
    has_persistent_location: bool
    persistent_location: Optional[str]
    persistent_entities_count: int
    conversation_depth: int
    time_since_last_turn: float
    contains_reference: bool
    follow_up_likelihood: float
    last_confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'has_previous_intent': self.has_previous_intent,
            'previous_intent': self.previous_intent,
            'intent_sequence': self.intent_sequence_pattern,
            'has_location': self.has_persistent_location,
            'location': self.persistent_location,
            'entities_count': self.persistent_entities_count,
            'depth': self.conversation_depth,
            'time_gap_sec': self.time_since_last_turn,
            'has_reference': self.contains_reference,
            'follow_up_score': self.follow_up_likelihood,
            'last_confidence': self.last_confidence
        }


# Common intent transition patterns (previous -> current)
INTENT_TRANSITION_PATTERNS = {
    'restaurant_query->transport_query': 0.85,      # "Find restaurant" -> "How to get there"
    'attraction_query->transport_query': 0.90,      # "Visit museum" -> "How to get there"
    'attraction_query->restaurant_query': 0.80,     # "Visit place" -> "Where to eat"
    'restaurant_query->restaurant_query': 0.75,     # "Find restaurant" -> "Similar restaurant"
    'attraction_query->attraction_query': 0.75,     # "Visit museum" -> "Another museum"
    'transport_query->restaurant_query': 0.70,      # "How to get" -> "Where to eat"
    'general_info->restaurant_query': 0.65,         # "Info" -> "Find restaurant"
    'general_info->attraction_query': 0.65,         # "Info" -> "Visit place"
}


class ContextAwareClassifier:
    """
    Enhanced intent classifier that uses conversation context
    """
    
    def __init__(self, context_manager: Optional[ConversationContextManager] = None):
        """Initialize context-aware classifier"""
        self.context_manager = context_manager or get_context_manager()
        
        # Confidence boost parameters
        self.same_intent_boost = 0.10
        self.pattern_match_boost = 0.08
        self.entity_context_boost = 0.05
        self.follow_up_boost = 0.07
        self.recent_location_boost = 0.04
        
        logger.info("✅ ContextAwareClassifier initialized")
    
    def extract_context_features(self, query: str, context: ConversationContext) -> ContextFeatures:
        """
        Extract features from conversation context
        
        Args:
            query: Current query
            context: Conversation context
            
        Returns:
            ContextFeatures object
        """
        # Get basic context info
        ctx_features = self.context_manager.extract_context_features(context)
        
        # Detect reference words in query
        contains_reference = self._contains_reference(query)
        
        # Calculate follow-up likelihood
        follow_up_likelihood = self._calculate_follow_up_likelihood(
            query, context, contains_reference
        )
        
        return ContextFeatures(
            has_previous_intent=ctx_features['has_previous_intent'],
            previous_intent=ctx_features['previous_intent'],
            intent_sequence_pattern=ctx_features['intent_sequence'],
            has_persistent_location=ctx_features['has_persistent_location'],
            persistent_location=ctx_features['persistent_location'],
            persistent_entities_count=ctx_features['persistent_entities_count'],
            conversation_depth=ctx_features['conversation_depth'],
            time_since_last_turn=ctx_features['time_since_last_turn'],
            contains_reference=contains_reference,
            follow_up_likelihood=follow_up_likelihood,
            last_confidence=ctx_features['last_confidence']
        )
    
    def _contains_reference(self, query: str) -> bool:
        """Check if query contains pronoun or implicit references"""
        query_lower = query.lower()
        
        # English reference words
        english_refs = [r'\bit\b', r'\bthere\b', r'\bthat\b', r'\bthose\b']
        
        # Turkish reference words
        turkish_refs = [r'\bora', r'\borgo', r'\bburası\b', r'\bşurası\b']
        
        # Implicit references
        implicit_refs = [
            r'^(ve|and)\s',  # Starting with "and"
            r'^(başka|else|other)',  # "anything else"
            r'^(daha|more)',  # "more options"
            r'^(yakın|nearby)',  # "nearby"
        ]
        
        all_patterns = english_refs + turkish_refs + implicit_refs
        
        return any(re.search(pattern, query_lower) for pattern in all_patterns)
    
    def _calculate_follow_up_likelihood(
        self,
        query: str,
        context: ConversationContext,
        contains_reference: bool
    ) -> float:
        """
        Calculate likelihood that query is a follow-up
        
        Returns:
            Float between 0 and 1
        """
        score = 0.0
        
        # Strong indicators
        if contains_reference:
            score += 0.40
        
        # Short queries are often follow-ups
        if len(query.split()) <= 3:
            score += 0.15
        
        # Recent conversation (within 30 seconds)
        last_turn = context.get_last_turn()
        if last_turn:
            time_gap = (context.updated_at - last_turn.timestamp).total_seconds()
            if time_gap < 30:
                score += 0.20
            elif time_gap < 120:
                score += 0.10
        
        # Multiple turns suggest ongoing conversation
        if context.get_conversation_depth() >= 2:
            score += 0.15
        
        # Has persistent entities
        if context.persistent_entities:
            score += 0.10
        
        return min(score, 1.0)
    
    def boost_confidence_with_context(
        self,
        base_confidence: float,
        intent: str,
        context: ConversationContext,
        context_features: ContextFeatures
    ) -> Tuple[float, Dict[str, float]]:
        """
        Boost classification confidence using context
        
        Args:
            base_confidence: Base classification confidence
            intent: Classified intent
            context: Conversation context
            context_features: Extracted context features
            
        Returns:
            Tuple of (boosted_confidence, boost_breakdown)
        """
        boost_breakdown = {}
        total_boost = 0.0
        
        # Boost if same intent as previous turn
        if context_features.has_previous_intent and context_features.previous_intent == intent:
            boost = self.same_intent_boost
            total_boost += boost
            boost_breakdown['same_intent'] = boost
            logger.debug(f"  + Same intent boost: +{boost:.3f}")
        
        # Boost if intent follows common transition pattern
        if context_features.has_previous_intent:
            pattern = f"{context_features.previous_intent}->{intent}"
            if pattern in INTENT_TRANSITION_PATTERNS:
                pattern_confidence = INTENT_TRANSITION_PATTERNS[pattern]
                boost = self.pattern_match_boost * pattern_confidence
                total_boost += boost
                boost_breakdown['pattern_match'] = boost
                logger.debug(f"  + Pattern match boost ({pattern}): +{boost:.3f}")
        
        # Boost if relevant entities exist in context
        relevant_entities = self.context_manager.get_relevant_entities(context, intent)
        if relevant_entities:
            boost = self.entity_context_boost * len(relevant_entities)
            total_boost += boost
            boost_breakdown['entity_context'] = boost
            logger.debug(f"  + Entity context boost ({len(relevant_entities)} entities): +{boost:.3f}")
        
        # Boost for high follow-up likelihood
        if context_features.follow_up_likelihood > 0.7:
            boost = self.follow_up_boost * context_features.follow_up_likelihood
            total_boost += boost
            boost_breakdown['follow_up'] = boost
            logger.debug(f"  + Follow-up boost: +{boost:.3f}")
        
        # Boost if query uses location from context
        if context_features.has_persistent_location and intent in ['restaurant_query', 'attraction_query']:
            boost = self.recent_location_boost
            total_boost += boost
            boost_breakdown['location_context'] = boost
            logger.debug(f"  + Location context boost: +{boost:.3f}")
        
        # Apply boost with ceiling at 0.99
        boosted_confidence = min(base_confidence + total_boost, 0.99)
        
        if total_boost > 0:
            logger.info(f"🚀 Confidence boosted: {base_confidence:.3f} → {boosted_confidence:.3f} (boost: +{total_boost:.3f})")
        
        return boosted_confidence, boost_breakdown
    
    def classify_with_context(
        self,
        query: str,
        preprocessed_query: str,
        entities: Dict[str, Any],
        base_intent: str,
        base_confidence: float,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Enhance classification result using conversation context
        
        Args:
            query: Original query
            preprocessed_query: Preprocessed query
            entities: Extracted entities
            base_intent: Base classification intent
            base_confidence: Base classification confidence
            session_id: Session identifier
            
        Returns:
            Enhanced classification result with context
        """
        # Get conversation context
        context = self.context_manager.get_context(session_id)
        
        # If no context, return base result
        if not context:
            logger.debug(f"No context for session {session_id}, using base classification")
            return {
                'intent': base_intent,
                'confidence': base_confidence,
                'original_confidence': base_confidence,
                'context_boost': 0.0,
                'boost_breakdown': {},
                'context_features': {},
                'resolved_query': query,
                'context_used': False
            }
        
        # Extract context features
        context_features = self.extract_context_features(query, context)
        
        logger.info(f"🧠 Context-aware classification for session {session_id}:")
        logger.info(f"  • Conversation depth: {context_features.conversation_depth}")
        logger.info(f"  • Previous intent: {context_features.previous_intent}")
        logger.info(f"  • Follow-up likelihood: {context_features.follow_up_likelihood:.2f}")
        logger.info(f"  • Contains reference: {context_features.contains_reference}")
        
        # Resolve references in query
        resolved_query = self.context_manager.resolve_references(query, context)
        
        # Boost confidence with context
        boosted_confidence, boost_breakdown = self.boost_confidence_with_context(
            base_confidence,
            base_intent,
            context,
            context_features
        )
        
        # Get relevant entities from context
        context_entities = self.context_manager.get_relevant_entities(context, base_intent)
        
        return {
            'intent': base_intent,
            'confidence': boosted_confidence,
            'original_confidence': base_confidence,
            'context_boost': boosted_confidence - base_confidence,
            'boost_breakdown': boost_breakdown,
            'context_features': context_features.to_dict(),
            'context_entities': context_entities,
            'resolved_query': resolved_query,
            'context_used': True,
            'conversation_depth': context_features.conversation_depth
        }
    
    def is_follow_up_query(self, query: str, session_id: str) -> bool:
        """
        Determine if a query is a follow-up to previous conversation
        
        Args:
            query: Query to check
            session_id: Session identifier
            
        Returns:
            True if query is likely a follow-up
        """
        context = self.context_manager.get_context(session_id)
        if not context:
            return False
        
        context_features = self.extract_context_features(query, context)
        return context_features.follow_up_likelihood > 0.6


# Singleton instance
_classifier_instance = None

def get_context_aware_classifier() -> ContextAwareClassifier:
    """Get or create context-aware classifier singleton"""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = ContextAwareClassifier()
    return _classifier_instance
