"""
Hybrid Intent Classifier
Combines neural (DistilBERT) and keyword-based classification for best accuracy

Strategy:
1. Try neural classifier first (GPU-accelerated)
2. Fall back to keyword matching if neural unavailable or low confidence
3. Ensemble: Combine scores when both agree for highest confidence
"""

import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass
from ..core.models import ConversationContext

logger = logging.getLogger(__name__)


@dataclass
class IntentResult:
    """Intent classification result"""
    primary_intent: str
    confidence: float = 0.0
    intents: list = None
    is_multi_intent: bool = False
    multi_intent_response: Optional[str] = None
    entities: Dict[str, Any] = None
    method: str = "keyword"  # 'neural', 'keyword', or 'ensemble'
    
    def __post_init__(self):
        if self.intents is None:
            self.intents = []
        if self.entities is None:
            self.entities = {}


class HybridIntentClassifier:
    """
    Hybrid intent classifier combining neural and keyword-based approaches
    
    Features:
    - GPU-accelerated neural classification (DistilBERT)
    - Keyword fallback for reliability
    - Ensemble scoring for maximum confidence
    - Automatic graceful degradation
    """
    
    def __init__(self, neural_classifier=None, keyword_classifier=None):
        """
        Initialize hybrid classifier
        
        Args:
            neural_classifier: NeuralQueryClassifier instance (optional)
            keyword_classifier: IntentClassifier instance (required)
        """
        self.neural = neural_classifier
        self.keyword = keyword_classifier
        self.use_neural = neural_classifier is not None
        
        # Statistics
        self.stats = {
            'neural_used': 0,
            'keyword_used': 0,
            'ensemble_used': 0,
            'neural_failures': 0
        }
        
        if self.use_neural:
            logger.info("✅ Hybrid classifier initialized (Neural + Keyword)")
        else:
            logger.info("⚠️  Hybrid classifier initialized (Keyword only - Neural unavailable)")
    
    def classify_intent(
        self,
        message: str,
        entities: Dict,
        context: Optional[ConversationContext] = None,
        neural_insights: Optional[Dict] = None,
        preprocessed_query: Optional[Any] = None,
        **kwargs
    ) -> IntentResult:
        """
        Classify intent using hybrid approach
        
        Args:
            message: User's input message
            entities: Extracted entities
            context: Conversation context (optional)
            neural_insights: Neural processing insights (optional)
            preprocessed_query: Preprocessed query data (optional)
            **kwargs: Additional arguments
        
        Returns:
            IntentResult with classification details
        """
        neural_result = None
        neural_intent = None
        neural_confidence = 0.0
        
        # STEP 1: Try neural classification (GPU-accelerated)
        if self.use_neural:
            try:
                # Call predict method (returns tuple: intent, confidence)
                neural_intent, neural_confidence = self.neural.predict(message)
                neural_result = True  # Mark that we got a result
                
                # If neural is highly confident, use it directly!
                if neural_confidence >= 0.80:
                    logger.info(f"🧠 Neural (high confidence): {neural_intent} ({neural_confidence:.2f})")
                    self.stats['neural_used'] += 1
                    
                    return IntentResult(
                        primary_intent=neural_intent,
                        confidence=neural_confidence,
                        intents=[neural_intent],
                        is_multi_intent=False,
                        entities=entities,
                        method='neural'
                    )
                
            except Exception as e:
                logger.warning(f"Neural classification failed: {e}")
                self.stats['neural_failures'] += 1
                neural_result = None
                neural_intent = None
                neural_confidence = 0.0
        
        # STEP 2: Get keyword classification (always run for ensemble/fallback)
        keyword_result = self.keyword.classify_intent(
            message=message,
            entities=entities,
            context=context,
            neural_insights=neural_insights,
            preprocessed_query=preprocessed_query
        )
        
        # STEP 3: Ensemble if both neural and keyword available
        if self.use_neural and neural_result is not None and neural_confidence >= 0.50:
            return self._ensemble_classification(
                neural_intent=neural_intent,
                neural_confidence=neural_confidence,
                keyword_result=keyword_result,
                entities=entities
            )
        
        # STEP 4: Pure keyword result (fallback)
        logger.debug(f"🔤 Keyword only: {keyword_result.primary_intent} ({keyword_result.confidence:.2f})")
        self.stats['keyword_used'] += 1
        keyword_result.method = 'keyword'
        return keyword_result
    
    def _ensemble_classification(
        self,
        neural_intent: str,
        neural_confidence: float,
        keyword_result: IntentResult,
        entities: Dict
    ) -> IntentResult:
        """
        Combine neural and keyword classifications using ensemble
        
        Args:
            neural_intent: Intent from neural classifier
            neural_confidence: Confidence from neural classifier
            keyword_result: Result from keyword classifier
            entities: Extracted entities
        
        Returns:
            IntentResult with ensemble classification
        """
        keyword_intent = keyword_result.primary_intent
        keyword_confidence = keyword_result.confidence
        
        # Case 1: Both agree - boost confidence!
        if neural_intent == keyword_intent:
            # Weighted average: 70% neural + 30% keyword + 10% agreement bonus
            final_confidence = min(
                neural_confidence * 0.7 + keyword_confidence * 0.3 + 0.10,
                1.0
            )
            logger.info(f"✅ Ensemble (agreement): {neural_intent} ({final_confidence:.2f})")
            self.stats['ensemble_used'] += 1
            
            return IntentResult(
                primary_intent=neural_intent,
                confidence=final_confidence,
                intents=[neural_intent],
                is_multi_intent=False,
                entities=entities,
                method='ensemble'
            )
        
        # Case 2: Disagreement - use higher confidence
        if neural_confidence >= keyword_confidence + 0.15:
            # Neural is significantly more confident
            logger.info(f"🧠 Neural (higher confidence): {neural_intent} ({neural_confidence:.2f}) vs keyword: {keyword_intent} ({keyword_confidence:.2f})")
            self.stats['neural_used'] += 1
            
            return IntentResult(
                primary_intent=neural_intent,
                confidence=neural_confidence,
                intents=[neural_intent, keyword_intent],
                is_multi_intent=True,
                entities=entities,
                method='neural'
            )
        
        elif keyword_confidence >= neural_confidence + 0.10:
            # Keyword is more confident
            logger.info(f"🔤 Keyword (higher confidence): {keyword_intent} ({keyword_confidence:.2f}) vs neural: {neural_intent} ({neural_confidence:.2f})")
            self.stats['keyword_used'] += 1
            
            return IntentResult(
                primary_intent=keyword_intent,
                confidence=keyword_confidence,
                intents=[keyword_intent, neural_intent],
                is_multi_intent=True,
                entities=entities,
                method='keyword'
            )
        
        # Case 3: Close call - use ensemble with reduced confidence
        final_confidence = (neural_confidence * 0.6 + keyword_confidence * 0.4) * 0.9  # Penalty for disagreement
        primary = neural_intent if neural_confidence > keyword_confidence else keyword_intent
        
        logger.info(f"⚖️  Ensemble (close call): {primary} ({final_confidence:.2f}) [neural: {neural_intent}, keyword: {keyword_intent}]")
        self.stats['ensemble_used'] += 1
        
        return IntentResult(
            primary_intent=primary,
            confidence=final_confidence,
            intents=[neural_intent, keyword_intent],
            is_multi_intent=True,
            entities=entities,
            method='ensemble'
        )
    
    def get_stats(self) -> Dict[str, int]:
        """Get classification statistics"""
        total = sum(self.stats.values()) - self.stats['neural_failures']
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'total_classifications': total,
            'neural_usage_pct': round(self.stats['neural_used'] / total * 100, 1),
            'keyword_usage_pct': round(self.stats['keyword_used'] / total * 100, 1),
            'ensemble_usage_pct': round(self.stats['ensemble_used'] / total * 100, 1)
        }
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'neural_used': 0,
            'keyword_used': 0,
            'ensemble_used': 0,
            'neural_failures': 0
        }
