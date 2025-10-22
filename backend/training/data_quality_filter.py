"""
Training Data Quality Filter
Filter user feedback to create high-quality training data
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict
from datetime import datetime
import logging
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class TrainingExample:
    """Represents a single training example"""
    
    def __init__(self, text: str, intent: str, language: str, 
                 confidence: float, source: str, metadata: Optional[Dict] = None):
        self.text = text
        self.intent = intent
        self.language = language
        self.confidence = confidence
        self.source = source
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'intent': self.intent,
            'language': self.language,
            'confidence': self.confidence,
            'source': self.source,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }


class TrainingDataQualityFilter:
    """
    Filter feedback data for training quality
    
    Quality Criteria:
    1. Explicit feedback preferred over implicit
    2. High confidence user feedback (not ambiguous)
    3. Diverse across all intent types
    4. No duplicate or near-duplicate queries
    5. Verified by multiple users or admin review
    """
    
    def __init__(self, 
                 min_explicit_confidence: float = 0.0,
                 min_implicit_confidence: float = 0.8,
                 similarity_threshold: float = 0.9,
                 min_samples_per_intent: int = 10,
                 max_samples_per_intent: int = 1000):
        """
        Initialize quality filter
        
        Args:
            min_explicit_confidence: Minimum confidence for explicit feedback (0.0 = accept all)
            min_implicit_confidence: Minimum confidence for implicit feedback
            similarity_threshold: Text similarity threshold for deduplication (0-1)
            min_samples_per_intent: Minimum samples required per intent
            max_samples_per_intent: Maximum samples to use per intent (for balance)
        """
        self.min_explicit_confidence = min_explicit_confidence
        self.min_implicit_confidence = min_implicit_confidence
        self.similarity_threshold = similarity_threshold
        self.min_samples_per_intent = min_samples_per_intent
        self.max_samples_per_intent = max_samples_per_intent
    
    def filter_training_data(self, feedback_samples: List[Any]) -> List[TrainingExample]:
        """
        Filter feedback samples to create high-quality training data
        
        Args:
            feedback_samples: List of IntentFeedback objects from database
            
        Returns:
            List of TrainingExample objects ready for model training
        """
        
        logger.info(f"🔍 Filtering {len(feedback_samples)} feedback samples...")
        
        # Step 1: Filter by quality threshold
        quality_samples = self._filter_by_quality(feedback_samples)
        logger.info(f"✅ {len(quality_samples)} samples passed quality filter")
        
        # Step 2: Deduplicate similar queries
        unique_samples = self._deduplicate_samples(quality_samples)
        logger.info(f"✅ {len(unique_samples)} unique samples after deduplication")
        
        # Step 3: Convert to training examples
        training_examples = self._convert_to_training_examples(unique_samples)
        logger.info(f"✅ {len(training_examples)} training examples created")
        
        # Step 4: Balance intent distribution
        balanced_examples = self._balance_intent_distribution(training_examples)
        logger.info(f"✅ {len(balanced_examples)} examples after balancing")
        
        # Step 5: Validate and report statistics
        self._report_statistics(balanced_examples)
        
        return balanced_examples
    
    def _filter_by_quality(self, feedback_samples: List[Any]) -> List[Any]:
        """Filter samples that meet quality thresholds"""
        quality_samples = []
        
        for sample in feedback_samples:
            if not self._meets_quality_threshold(sample):
                continue
            
            quality_samples.append(sample)
        
        return quality_samples
    
    def _meets_quality_threshold(self, sample: Any) -> bool:
        """
        Check if sample meets minimum quality standards
        
        Quality checks:
        - Explicit feedback with clear intent (correct or corrected)
        - Implicit feedback with high confidence
        - Not marked as spam or low quality
        - Has clear text and intent
        """
        
        # Must have text and intent
        if not sample.original_query or not sample.original_query.strip():
            return False
        
        # Get the actual intent (corrected or predicted)
        intent = sample.actual_intent if sample.actual_intent else sample.predicted_intent
        if not intent:
            return False
        
        # Explicit feedback is higher quality
        if sample.feedback_type == 'explicit':
            # Must have clear feedback (correct or incorrect with correction)
            if sample.is_correct is None:
                return False
            
            # If incorrect, must have correction
            if sample.is_correct is False and not sample.actual_intent:
                return False
            
            # Check confidence threshold
            if sample.predicted_confidence < self.min_explicit_confidence:
                return False
            
            return True
        
        # Implicit feedback needs high confidence
        if sample.feedback_type == 'implicit':
            # Must have inferred correctness
            if sample.is_correct is None:
                return False
            
            # Only accept positive implicit feedback with high confidence
            if not sample.is_correct:
                return False
            
            if sample.predicted_confidence < self.min_implicit_confidence:
                return False
            
            return True
        
        return False
    
    def _deduplicate_samples(self, samples: List[Any]) -> List[Any]:
        """
        Remove duplicate and near-duplicate queries
        
        Uses text similarity to find duplicates while preserving
        different phrasings of the same intent
        """
        
        unique_samples = []
        seen_texts = []
        
        for sample in samples:
            text = sample.original_query.lower().strip()
            
            # Check if similar text already exists
            is_duplicate = False
            for seen_text in seen_texts:
                similarity = self._text_similarity(text, seen_text)
                if similarity >= self.similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_samples.append(sample)
                seen_texts.append(text)
        
        return unique_samples
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity using SequenceMatcher
        
        Returns similarity score between 0 and 1
        """
        return SequenceMatcher(None, text1, text2).ratio()
    
    def _convert_to_training_examples(self, samples: List[Any]) -> List[TrainingExample]:
        """Convert feedback samples to training examples"""
        
        examples = []
        
        for sample in samples:
            # Use corrected intent if available, otherwise predicted
            intent = sample.actual_intent if sample.actual_intent else sample.predicted_intent
            
            # Calculate confidence for this training example
            confidence = self._calculate_sample_confidence(sample)
            
            example = TrainingExample(
                text=sample.original_query,
                intent=intent,
                language=sample.language or 'unknown',
                confidence=confidence,
                source='user_feedback',
                metadata={
                    'feedback_id': sample.id,
                    'feedback_type': sample.feedback_type,
                    'is_correct': sample.is_correct,
                    'original_confidence': sample.predicted_confidence,
                    'session_id': sample.session_id,
                    'timestamp': sample.timestamp.isoformat() if sample.timestamp else None
                }
            )
            
            examples.append(example)
        
        return examples
    
    def _calculate_sample_confidence(self, sample: Any) -> float:
        """
        Calculate confidence score for training example
        
        Higher confidence = more reliable training data
        """
        
        base_confidence = 0.5
        
        # Explicit feedback is more reliable
        if sample.feedback_type == 'explicit':
            if sample.is_correct:
                base_confidence = 0.9  # User confirmed correct
            else:
                base_confidence = 0.95  # User provided correction (very valuable)
        
        # Implicit positive feedback with high model confidence
        elif sample.feedback_type == 'implicit':
            if sample.is_correct and sample.predicted_confidence > 0.9:
                base_confidence = 0.85
            elif sample.is_correct:
                base_confidence = 0.75
        
        # Boost if reviewed by admin
        if sample.review_status == 'approved':
            base_confidence = min(1.0, base_confidence + 0.05)
        
        return base_confidence
    
    def _balance_intent_distribution(self, examples: List[TrainingExample]) -> List[TrainingExample]:
        """
        Balance training data across intents
        
        Strategy:
        - Ensure minimum samples per intent
        - Cap maximum samples per intent
        - Preserve diversity within each intent
        """
        
        # Group by intent
        intent_groups = defaultdict(list)
        for example in examples:
            intent_groups[example.intent].append(example)
        
        balanced = []
        
        for intent, intent_examples in intent_groups.items():
            # Skip intents with too few samples
            if len(intent_examples) < self.min_samples_per_intent:
                logger.warning(
                    f"⚠️ Intent '{intent}' has only {len(intent_examples)} samples "
                    f"(minimum: {self.min_samples_per_intent}). Skipping."
                )
                continue
            
            # Cap at maximum samples per intent
            if len(intent_examples) > self.max_samples_per_intent:
                # Sort by confidence (descending) to keep best samples
                intent_examples.sort(key=lambda x: x.confidence, reverse=True)
                intent_examples = intent_examples[:self.max_samples_per_intent]
                logger.info(
                    f"📊 Intent '{intent}' capped at {self.max_samples_per_intent} samples"
                )
            
            balanced.extend(intent_examples)
        
        return balanced
    
    def _report_statistics(self, examples: List[TrainingExample]):
        """Report statistics about filtered training data"""
        
        if not examples:
            logger.warning("⚠️ No training examples generated!")
            return
        
        # Intent distribution
        intent_counts = Counter(ex.intent for ex in examples)
        
        # Language distribution
        language_counts = Counter(ex.language for ex in examples)
        
        # Source distribution
        source_counts = Counter(ex.source for ex in examples)
        
        # Average confidence
        avg_confidence = sum(ex.confidence for ex in examples) / len(examples)
        
        logger.info("\n" + "=" * 60)
        logger.info("Training Data Quality Report")
        logger.info("=" * 60)
        logger.info(f"Total Examples: {len(examples)}")
        logger.info(f"Average Confidence: {avg_confidence:.2%}")
        
        logger.info("\n📊 Intent Distribution:")
        for intent, count in intent_counts.most_common():
            percentage = (count / len(examples)) * 100
            logger.info(f"  {intent:20s}: {count:4d} ({percentage:5.1f}%)")
        
        logger.info("\n🌍 Language Distribution:")
        for lang, count in language_counts.most_common():
            percentage = (count / len(examples)) * 100
            logger.info(f"  {lang:10s}: {count:4d} ({percentage:5.1f}%)")
        
        logger.info("\n📁 Source Distribution:")
        for source, count in source_counts.most_common():
            percentage = (count / len(examples)) * 100
            logger.info(f"  {source:15s}: {count:4d} ({percentage:5.1f}%)")
        
        logger.info("=" * 60 + "\n")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Training Data Quality Filter - Test")
    print("=" * 60)
    
    # Create mock feedback samples
    class MockFeedback:
        def __init__(self, query, intent, confidence, feedback_type, is_correct, 
                     actual_intent=None, review_status='pending'):
            self.id = hash(query)
            self.original_query = query
            self.predicted_intent = intent
            self.actual_intent = actual_intent
            self.predicted_confidence = confidence
            self.feedback_type = feedback_type
            self.is_correct = is_correct
            self.review_status = review_status
            self.language = 'tr' if any(c in query for c in 'çğıöşüÇĞİÖŞÜ') else 'en'
            self.session_id = 'test_session'
            self.timestamp = datetime.utcnow()
    
    mock_samples = [
        MockFeedback("Sultanahmet'te restoran", "restaurant", 0.85, "explicit", True),
        MockFeedback("Best museums in Istanbul", "museum", 0.92, "explicit", True),
        MockFeedback("Taksi nasıl çağırırım", "transportation", 0.78, "implicit", True),
        MockFeedback("Show me attractions", "museum", 0.65, "explicit", False, "attraction"),
        MockFeedback("Sultanahmet'te restoran öner", "restaurant", 0.88, "implicit", True),  # Near duplicate
        MockFeedback("Ayasofya nerede", "attraction", 0.91, "explicit", True),
    ]
    
    # Create filter
    filter_system = TrainingDataQualityFilter(
        min_implicit_confidence=0.75,
        similarity_threshold=0.85
    )
    
    # Filter data
    training_examples = filter_system.filter_training_data(mock_samples)
    
    print(f"\n✅ Generated {len(training_examples)} training examples")
    print("\nSample examples:")
    for i, example in enumerate(training_examples[:3], 1):
        print(f"\n{i}. Text: {example.text}")
        print(f"   Intent: {example.intent}")
        print(f"   Language: {example.language}")
        print(f"   Confidence: {example.confidence:.2%}")
