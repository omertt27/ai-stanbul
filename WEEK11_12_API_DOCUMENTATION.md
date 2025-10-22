# API Documentation - Enhanced Intent Classification System

**Version:** 2.0  
**Last Updated:** October 22, 2025  
**Status:** Production Ready

---

## 📚 Overview

Complete API documentation for the enhanced Istanbul AI intent classification system, including preprocessing pipeline, context-aware classification, and entity extraction.

---

## 🔧 System Components

### 1. Query Preprocessing Pipeline

**Purpose:** Normalizes queries through typo correction, dialect normalization, and entity extraction.

#### API

```python
from backend.services.query_preprocessing_pipeline import QueryPreprocessingPipeline

pipeline = QueryPreprocessingPipeline()
result = pipeline.process(query, intent=None)
```

#### Input Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | Raw user query |
| `intent` | string | No | Optional intent hint for entity extraction |

#### Response Object

```python
@dataclass
class QueryProcessingResult:
    original_query: str              # Original input
    cleaned_query: str               # Processed query
    entities: Dict[str, Any]         # Extracted entities
    intent: Optional[str]            # Detected intent
    
    # Processing details
    typo_corrections: list           # Applied corrections
    dialect_normalizations: list     # Applied normalizations
    has_typos: bool                 # Typos detected
    has_dialect: bool               # Dialect detected
    
    # Performance metrics
    typo_correction_ms: float       # Time taken
    dialect_normalization_ms: float # Time taken
    entity_extraction_ms: float     # Time taken
    total_processing_ms: float      # Total time
```

#### Example

```python
# Basic usage
result = pipeline.process("sultanahmete nasıl gidilr")

print(result.cleaned_query)  # "sultanahmet'e nasıl gidilir"
print(result.has_typos)      # True
print(result.entities)       # {'locations': ['Sultanahmet'], ...}
```

#### Performance

- **Average Latency:** 0.08ms
- **P95 Latency:** 0.13ms
- **Throughput:** 8,800+ req/s

---

### 2. Context-Aware Classifier

**Purpose:** Enhances intent classification using conversation context and history.

#### API

```python
from backend.services.context_aware_classifier import ContextAwareClassifier
from backend.services.conversation_context_manager import ConversationContextManager

context_manager = ConversationContextManager()
classifier = ContextAwareClassifier(context_manager)

result = classifier.classify_with_context(
    query=query,
    preprocessed_query=preprocessed,
    base_intent=neural_intent,
    base_confidence=neural_conf,
    session_id=session_id,
    entities=entities
)
```

#### Input Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | Original user query |
| `preprocessed_query` | string | Yes | Preprocessed query |
| `base_intent` | string | Yes | Base intent from neural classifier |
| `base_confidence` | float | Yes | Base confidence (0-1) |
| `session_id` | string | Yes | User session identifier |
| `entities` | dict | No | Extracted entities |

#### Response Object

```python
{
    'intent': str,                    # Final intent
    'confidence': float,              # Boosted confidence
    'original_confidence': float,     # Original confidence
    'context_boost': float,           # Applied boost
    'is_follow_up': bool,            # Is follow-up query
    'context_features': dict,         # Context features used
    'reasoning': str                  # Explanation
}
```

#### Example

```python
# First turn
result1 = classifier.classify_with_context(
    query="ayasofya'yı görmek istiyorum",
    preprocessed_query="ayasofya'yı görmek istiyorum",
    base_intent="places_attractions",
    base_confidence=0.8,
    session_id="user123",
    entities={'locations': ['Ayasofya']}
)

# Update context
from backend.services.conversation_context_manager import Turn
turn = Turn(
    query=result1['query'],
    preprocessed_query=result1['preprocessed_query'],
    intent=result1['intent'],
    entities=entities,
    confidence=result1['confidence']
)
context_manager.update_context("user123", turn)

# Follow-up turn
result2 = classifier.classify_with_context(
    query="oraya nasıl giderim",  # "how do I get there"
    preprocessed_query="oraya nasıl giderim",
    base_intent="transportation",
    base_confidence=0.7,
    session_id="user123",
    entities={}
)

print(result2['is_follow_up'])     # True
print(result2['context_boost'])    # +0.15 (context boost)
print(result2['confidence'])       # 0.85 (boosted)
```

#### Performance

- **Average Latency:** 0.05ms
- **P95 Latency:** 0.08ms
- **Context Boost:** +10% to +30%

---

### 3. Conversation Context Manager

**Purpose:** Tracks and manages conversation history and context.

#### API

```python
from backend.services.conversation_context_manager import (
    ConversationContextManager, 
    Turn
)

context_manager = ConversationContextManager(
    redis_host='localhost',
    redis_port=6379,
    redis_db=1
)
```

#### Key Methods

##### Update Context

```python
turn = Turn(
    query="sultanahmet'e nasıl gidilir",
    preprocessed_query="sultanahmet'e nasıl gidilir",
    intent="transportation",
    entities={'locations': ['Sultanahmet']},
    confidence=0.85
)

success = context_manager.update_context(session_id, turn)
```

##### Get Context

```python
context = context_manager.get_context(session_id)

# Returns ConversationContext object with:
# - conversation_history: List[Turn]
# - persistent_entities: Dict
# - intent_history: List[str]
# - last_location, last_restaurant, etc.
```

##### Clear Context

```python
context_manager.clear_context(session_id)
```

#### Storage

- **Primary:** Redis (if available)
- **Fallback:** In-memory dictionary
- **TTL:** 24 hours (configurable)
- **Sliding Window:** 5 turns (configurable)

---

### 4. Dynamic Threshold Manager

**Purpose:** Manages adaptive confidence thresholds per intent.

#### API

```python
from backend.services.dynamic_threshold_manager import DynamicThresholdManager

threshold_manager = DynamicThresholdManager()

# Get threshold for intent
threshold = threshold_manager.get_threshold("transportation")

# Check if should accept
should_accept, decision = threshold_manager.should_accept(
    intent="transportation",
    confidence=0.75,
    context_features=context_features,
    entities=entities
)

# Record outcome
threshold_manager.record_acceptance(
    intent="transportation",
    confidence=0.75,
    accepted=True
)
```

#### Threshold Adjustments

| Factor | Adjustment | Condition |
|--------|-----------|-----------|
| **High Entity Confidence** | -5% | Entities extracted with high confidence |
| **Recent Same Intent** | -10% | Same intent in last 3 turns |
| **Follow-up Query** | -15% | Detected follow-up |
| **Low Ambiguity** | -5% | Single clear intent |
| **Multiple Same Intents** | -5% | Intent repeated in history |
| **High Context Relevance** | -10% | Strong context match |
| **Consecutive Same Intent** | -15% | 2+ consecutive same intent |

---

## 🔗 Backend Integration

### Complete Flow Example

```python
from backend.services.query_preprocessing_pipeline import QueryPreprocessingPipeline
from backend.services.context_aware_classifier import ContextAwareClassifier
from backend.services.conversation_context_manager import (
    ConversationContextManager, Turn
)
from backend.services.dynamic_threshold_manager import DynamicThresholdManager

# Initialize components
preprocessing = QueryPreprocessingPipeline()
context_mgr = ConversationContextManager()
classifier = ContextAwareClassifier(context_mgr)
threshold_mgr = DynamicThresholdManager()

# Process query
def process_query(query, session_id):
    # Step 1: Preprocess
    preprocessed = preprocessing.process(query)
    
    # Step 2: Neural classification (your existing classifier)
    neural_result = your_neural_classifier(preprocessed.cleaned_query)
    
    # Step 3: Context-aware classification
    result = classifier.classify_with_context(
        query=query,
        preprocessed_query=preprocessed.cleaned_query,
        base_intent=neural_result['intent'],
        base_confidence=neural_result['confidence'],
        session_id=session_id,
        entities=preprocessed.entities
    )
    
    # Step 4: Check threshold
    context = context_mgr.get_context(session_id)
    context_features = classifier.extract_context_features(query, context)
    
    should_accept, decision = threshold_mgr.should_accept(
        intent=result['intent'],
        confidence=result['confidence'],
        context_features=context_features,
        entities=preprocessed.entities
    )
    
    # Step 5: Update context if accepted
    if should_accept:
        turn = Turn(
            query=query,
            preprocessed_query=preprocessed.cleaned_query,
            intent=result['intent'],
            entities=preprocessed.entities,
            confidence=result['confidence']
        )
        context_mgr.update_context(session_id, turn)
    
    return {
        'intent': result['intent'],
        'confidence': result['confidence'],
        'entities': preprocessed.entities,
        'accepted': should_accept,
        'preprocessing_time_ms': preprocessed.total_processing_ms,
        'context_boost': result['context_boost']
    }
```

---

## 📊 Performance Metrics

### System-Wide Performance

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **End-to-End Latency (avg)** | 0.12ms | <200ms | ✅ 1,667x better |
| **End-to-End Latency (P95)** | 0.18ms | - | ✅ |
| **End-to-End Latency (P99)** | 0.22ms | - | ✅ |
| **Throughput (single)** | 8,820 req/s | >100 req/s | ✅ 88x better |
| **Throughput (concurrent)** | 7,619 req/s | >500 req/s | ✅ 15x better |
| **Memory Usage** | 0.13 MB | <512 MB | ✅ 3,938x better |

### Component Performance

| Component | Average | P95 | P99 |
|-----------|---------|-----|-----|
| **Preprocessing** | 0.08ms | 0.13ms | 0.14ms |
| **Context-Aware** | 0.05ms | 0.08ms | 0.11ms |
| **Entity Extraction** | Included in preprocessing | - | - |

---

## 🔒 Error Handling

### Graceful Degradation

The system is designed to degrade gracefully:

1. **Redis Unavailable:** Falls back to in-memory context storage
2. **Entity Extraction Fails:** Continues with empty entities
3. **Invalid Confidence:** Clamps to [0, 1] range
4. **Context Errors:** Uses base classification without context boost

### Error Responses

```python
try:
    result = process_query(query, session_id)
except Exception as e:
    logger.error(f"Query processing error: {e}")
    # Return base classification without enhancements
    result = {
        'intent': 'unknown',
        'confidence': 0.0,
        'error': str(e)
    }
```

---

## 🧪 Testing

### Run Integration Tests

```bash
python test_week11_integration.py
```

### Run Performance Benchmarks

```bash
python test_performance_benchmarks.py
```

### Run Specific Component Tests

```bash
# Entity extraction
python test_entity_extractor.py

# Dialect/typo correction
python test_dialect_typo.py

# Context-aware system
python test_context_aware_system.py
```

---

## 📈 Monitoring

### Key Metrics to Track

1. **Latency Metrics:**
   - Preprocessing time
   - Context-aware classification time
   - End-to-end response time

2. **Accuracy Metrics:**
   - Intent classification accuracy
   - Entity extraction accuracy
   - Context boost effectiveness

3. **System Health:**
   - Redis connection status
   - Memory usage
   - Error rates

### Logging

```python
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# All components log performance and errors
```

---

## 🔄 Configuration

### Environment Variables

```bash
# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=1
REDIS_TTL=86400  # 24 hours

# Context Configuration
CONTEXT_SLIDING_WINDOW=5
CONTEXT_MAX_HISTORY=10

# Threshold Configuration
DEFAULT_THRESHOLD=0.7
ENABLE_ADAPTIVE_THRESHOLDS=true
```

---

## 📞 Support

For issues or questions:
- Check test files for usage examples
- Review component documentation
- See WEEK11_12_PHASE1_COMPLETE.md for performance details

---

**Document Version:** 2.0  
**API Version:** 2.0  
**Last Updated:** October 22, 2025
