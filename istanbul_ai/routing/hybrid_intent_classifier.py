"""
Hybrid Intent Classifier
Combines neural (DistilBERT) and keyword-based classification for best accuracy

Strategy:
1. Try neural classifier first (GPU-accelerated)
2. Fall back to keyword matching if neural unavailable or low confidence
3. Ensemble: Combine scores when both agree for highest confidence
"""

import logging
import os
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
    
    def __init__(self, neural_classifier=None, keyword_classifier=None, llm_classifier=None):
        """
        Initialize hybrid classifier
        
        Args:
            neural_classifier: NeuralQueryClassifier instance (optional)
            keyword_classifier: IntentClassifier instance (required)
            llm_classifier: LLMIntentClassifier for low-confidence fallback (optional)
        """
        self.neural = neural_classifier
        self.keyword = keyword_classifier
        self.llm = llm_classifier
        self.use_neural = neural_classifier is not None
        self.use_llm = llm_classifier is not None
        
        # Confidence threshold for LLM fallback
        self.llm_fallback_threshold = float(os.getenv("LLM_FALLBACK_THRESHOLD", 0.60))  # If confidence < threshold, try LLM
        
        # Enhanced statistics with performance metrics
        self.stats = {
            'neural_used': 0,
            'keyword_used': 0,
            'ensemble_used': 0,
            'llm_used': 0,
            'neural_failures': 0,
            'llm_failures': 0,
            'llm_improved_count': 0,  # Track when LLM improved confidence
            'confidence_distribution': {  # Track confidence ranges
                '0.0-0.2': 0,
                '0.2-0.4': 0,
                '0.4-0.6': 0,
                '0.6-0.8': 0,
                '0.8-1.0': 0
            }
        }
        
        classifier_modes = []
        if self.use_neural:
            classifier_modes.append("Neural")
        classifier_modes.append("Keyword")
        if self.use_llm:
            classifier_modes.append("LLM")
        
        logger.info(f"✅ Hybrid classifier initialized ({' + '.join(classifier_modes)})")
    
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
                
                # CRITICAL: Check for transportation keywords BEFORE accepting neural result
                # This prevents neural misclassification of obvious transportation queries
                message_lower = message.lower()
                strong_transportation_keywords = [
                    'how can i go', 'how do i get', 'how to get to', 'how to go to',
                    'directions to', 'navigate to', 'route to', 'way to get',
                    'take me to', 'show me the way', 'from my location', 'from here to',
                    'metro to', 'bus to', 'tram to', 'transportation to',
                    'how far is', 'distance to', 'travel to'
                ]
                
                has_strong_transportation = any(keyword in message_lower for keyword in strong_transportation_keywords)
                
                # If neural is highly confident BUT it's clearly a transportation query, override it
                if neural_confidence >= 0.80:
                    if has_strong_transportation and neural_intent != 'transportation':
                        logger.warning(f"⚠️ Neural misclassified transportation query as '{neural_intent}' - forcing transportation")
                        self.stats['keyword_used'] += 1
                        return IntentResult(
                            primary_intent='transportation',
                            confidence=0.95,  # High confidence for keyword override
                            intents=['transportation'],
                            is_multi_intent=False,
                            entities=entities,
                            method='keyword_override'
                        )
                    
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
        
        # Track confidence distribution
        self._track_confidence(keyword_result.confidence)
        
        # STEP 5: LLM fallback if confidence is low
        if self.use_llm and keyword_result.confidence < self.llm_fallback_threshold:
            logger.info(f"⚠️ Low confidence ({keyword_result.confidence:.2f} < {self.llm_fallback_threshold}) - trying LLM for better understanding")
            try:
                llm_result = self.llm.classify_intent(
                    message=message,
                    entities=entities,
                    context=context,
                    neural_insights=neural_insights,
                    preprocessed_query=preprocessed_query
                )
                
                if llm_result and llm_result.confidence > keyword_result.confidence:
                    improvement = llm_result.confidence - keyword_result.confidence
                    logger.info(f"✅ LLM improved classification: {llm_result.primary_intent} ({llm_result.confidence:.2f}) +{improvement:.2f}")
                    self.stats['llm_used'] += 1
                    self.stats['llm_improved_count'] += 1
                    self._track_confidence(llm_result.confidence)
                    llm_result.method = 'llm_fallback'
                    return llm_result
                else:
                    logger.debug("LLM didn't improve confidence, using keyword result")
                    
            except Exception as e:
                logger.warning(f"❌ LLM fallback failed: {e}")
                self.stats['llm_failures'] += 1
        
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
        if neural_confidence >= keyword_confidence + 0.10:
            # Neural is significantly more confident
            logger.info(f"🧠 Neural (higher confidence): {neural_intent} ({neural_confidence:.2f}) vs keyword: {keyword_intent} ({keyword_confidence:.2f})")
            self.stats['neural_used'] += 1
            
            return IntentResult(
                primary_intent=neural_intent,
                confidence=neural_confidence,
                intents=[neural_intent],
                is_multi_intent=False,
                entities=entities,
                method='neural'
            )
        
        elif keyword_confidence >= neural_confidence + 0.20:
            # Keyword is significantly more confident (increased threshold from 0.10 to 0.20)
            logger.info(f"🔤 Keyword (higher confidence): {keyword_intent} ({keyword_confidence:.2f}) vs neural: {neural_intent} ({neural_confidence:.2f})")
            self.stats['keyword_used'] += 1
            
            return IntentResult(
                primary_intent=keyword_intent,
                confidence=keyword_confidence,
                intents=[keyword_intent],
                is_multi_intent=False,
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
    
    def _track_confidence(self, confidence: float):
        """Track confidence distribution for analytics"""
        if confidence < 0.2:
            self.stats['confidence_distribution']['0.0-0.2'] += 1
        elif confidence < 0.4:
            self.stats['confidence_distribution']['0.2-0.4'] += 1
        elif confidence < 0.6:
            self.stats['confidence_distribution']['0.4-0.6'] += 1
        elif confidence < 0.8:
            self.stats['confidence_distribution']['0.6-0.8'] += 1
        else:
            self.stats['confidence_distribution']['0.8-1.0'] += 1
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        total = (self.stats['neural_used'] + self.stats['keyword_used'] + 
                self.stats['ensemble_used'] + self.stats['llm_used'])
        
        if total == 0:
            total = 1  # Avoid division by zero
        
        return {
            'total_requests': total,
            'neural_count': self.stats['neural_used'],
            'keyword_count': self.stats['keyword_used'],
            'ensemble_count': self.stats['ensemble_used'],
            'llm_fallback_count': self.stats['llm_used'],
            'llm_fallback_rate': (self.stats['llm_used'] / total) * 100,
            'llm_improved_count': self.stats['llm_improved_count'],
            'llm_improvement_rate': (self.stats['llm_improved_count'] / max(self.stats['llm_used'], 1)) * 100,
            'neural_failures': self.stats['neural_failures'],
            'llm_failures': self.stats['llm_failures'],
            'confidence_distribution': dict(self.stats['confidence_distribution'])
        }
