# Week 9-10: Context-Aware Classification - IMPLEMENTATION PLAN 🎯

**Start Date:** October 23, 2025  
**Target Completion:** November 6, 2025  
**Status:** 🚀 READY TO START

---

## 🎯 Phase Overview

### Objectives
Implement context-aware query classification to improve accuracy for follow-up queries and maintain conversation continuity.

### Key Goals
1. **Conversation Context Manager** - Track conversation history and extract context features
2. **Context-Aware Intent Classifier** - Use context to boost classification confidence
3. **Follow-up Query Resolution** - Handle references to previous conversation turns
4. **Dynamic Threshold Management** - Adjust confidence thresholds based on context

---

## 📋 Detailed Components

### Component 1: Conversation Context Manager

#### Features
- **Session-based context storage** using Redis
- **Sliding window** (configurable, default: 5 turns)
- **Entity persistence** across conversation
- **Intent history** tracking
- **User preference** learning

#### Data Structure
```python
ConversationContext:
    - session_id: str
    - conversation_history: List[Turn]  # Last N turns
    - persistent_entities: Dict[str, Any]  # Entities mentioned
    - intent_history: List[str]  # Previous intents
    - user_preferences: Dict[str, Any]  # Learned preferences
    - last_location: Optional[str]  # Last mentioned location
    - last_restaurant: Optional[str]  # Last mentioned restaurant
    - last_museum: Optional[str]  # Last mentioned museum
    - timestamp: datetime
```

#### Turn Structure
```python
Turn:
    - query: str  # Original query
    - preprocessed_query: str  # After preprocessing
    - intent: str  # Classified intent
    - entities: Dict[str, Any]  # Extracted entities
    - confidence: float  # Classification confidence
    - timestamp: datetime
```

#### API
```python
class ConversationContextManager:
    def get_context(session_id: str) -> ConversationContext
    def update_context(session_id: str, turn: Turn) -> None
    def extract_context_features(context: ConversationContext) -> Dict
    def resolve_references(query: str, context: ConversationContext) -> str
    def get_relevant_entities(context: ConversationContext) -> Dict
    def clear_context(session_id: str) -> None
```

---

### Component 2: Context-Aware Intent Classifier

#### Features
- **Context feature extraction** from conversation history
- **Confidence boosting** for related intents
- **Reference resolution** ("it", "there", "that place")
- **Intent transition patterns** (common sequences)

#### Context Features
```python
ContextFeatures:
    - has_previous_intent: bool
    - previous_intent: Optional[str]
    - intent_sequence_pattern: Optional[str]  # e.g., "restaurant->transport"
    - has_persistent_location: bool
    - persistent_entities_count: int
    - conversation_depth: int  # Number of turns
    - time_since_last_turn: float  # Seconds
    - contains_reference: bool  # "it", "there", "that"
    - follow_up_likelihood: float  # 0-1
```

#### Enhanced Classification Logic
```python
def classify_with_context(
    query: str,
    preprocessed_query: str,
    entities: Dict,
    context: ConversationContext
) -> ClassificationResult:
    # Extract context features
    context_features = extract_context_features(context)
    
    # Resolve references
    resolved_query = resolve_references(query, context)
    
    # Base classification
    base_result = base_classifier.classify(resolved_query)
    
    # Context-aware confidence boosting
    if is_follow_up(query, context_features):
        boosted_result = boost_confidence(base_result, context)
        return boosted_result
    
    return base_result
```

---

### Component 3: Follow-up Query Resolution

#### Reference Types to Handle

**Pronoun References:**
- "it" → last mentioned entity
- "there" → last mentioned location
- "that place" → last mentioned location
- "them" → multiple entities

**Implicit References:**
- "what about lunch?" → current location context
- "how do I get there?" → last mentioned place
- "is it open now?" → last mentioned venue

**Continuation Patterns:**
- "and what else?" → same intent, different results
- "anything nearby?" → location from context
- "any cheaper options?" → price preference change

#### Resolution Algorithm
```python
def resolve_references(query: str, context: ConversationContext) -> str:
    """Resolve pronouns and implicit references"""
    
    resolved = query
    
    # Detect reference words
    if has_pronoun_reference(query):
        # "it" → last entity
        if "it" in query.lower():
            if context.last_restaurant:
                resolved = resolved.replace("it", context.last_restaurant)
            elif context.last_museum:
                resolved = resolved.replace("it", context.last_museum)
        
        # "there" → last location
        if "there" in query.lower():
            if context.last_location:
                resolved = resolved.replace("there", context.last_location)
    
    # Add context entities if missing
    if needs_location_context(query) and not has_location(query):
        if context.last_location:
            resolved = f"{resolved} in {context.last_location}"
    
    return resolved
```

---

### Component 4: Dynamic Threshold Manager

#### Intent-Specific Thresholds
```python
THRESHOLD_CONFIG = {
    "restaurant_query": {
        "base": 0.70,
        "with_context": 0.60,  # Lower threshold for follow-ups
        "with_entities": 0.65
    },
    "attraction_query": {
        "base": 0.70,
        "with_context": 0.60,
        "with_entities": 0.65
    },
    "transport_query": {
        "base": 0.75,  # Higher threshold (more specific)
        "with_context": 0.65,
        "with_entities": 0.70
    },
    "general_info": {
        "base": 0.50,  # Lower threshold (catch-all)
        "with_context": 0.40,
        "with_entities": 0.45
    }
}
```

#### Adaptive Threshold Logic
```python
def get_adaptive_threshold(
    intent: str,
    context_features: ContextFeatures,
    entities: Dict
) -> float:
    """Calculate adaptive confidence threshold"""
    
    config = THRESHOLD_CONFIG.get(intent, {"base": 0.70})
    
    # Start with base threshold
    threshold = config["base"]
    
    # Adjust based on context
    if context_features.has_previous_intent:
        threshold = config.get("with_context", threshold - 0.10)
    
    # Adjust based on entities
    if len(entities) >= 2:
        threshold = config.get("with_entities", threshold - 0.05)
    
    # Adjust for follow-up queries
    if context_features.follow_up_likelihood > 0.7:
        threshold *= 0.9  # 10% reduction
    
    return threshold
```

---

## 🏗️ Implementation Roadmap

### Phase 1: Context Manager (Days 1-3)

**Day 1: Core Infrastructure**
- [ ] Create `conversation_context_manager.py`
- [ ] Define data structures (ConversationContext, Turn)
- [ ] Implement Redis-based storage
- [ ] Add session management

**Day 2: Context Operations**
- [ ] Implement context retrieval
- [ ] Implement context updates
- [ ] Add sliding window logic
- [ ] Implement entity persistence

**Day 3: Feature Extraction**
- [ ] Implement context feature extraction
- [ ] Add reference detection
- [ ] Implement entity aggregation
- [ ] Add unit tests

---

### Phase 2: Context-Aware Classifier (Days 4-6)

**Day 4: Enhanced Classification**
- [ ] Create `context_aware_classifier.py`
- [ ] Implement context feature extraction
- [ ] Add reference resolution logic
- [ ] Integrate with existing classifier

**Day 5: Confidence Boosting**
- [ ] Implement confidence boosting algorithm
- [ ] Add intent transition patterns
- [ ] Implement follow-up detection
- [ ] Add validation logic

**Day 6: Integration**
- [ ] Update main classifier to use context
- [ ] Integrate with preprocessing pipeline
- [ ] Add context to API responses
- [ ] Create integration tests

---

### Phase 3: Follow-up Resolution (Days 7-9)

**Day 7: Reference Resolution**
- [ ] Implement pronoun resolution
- [ ] Add location context resolution
- [ ] Implement entity substitution
- [ ] Add disambiguation logic

**Day 8: Implicit Reference Handling**
- [ ] Detect implicit references
- [ ] Add context entity injection
- [ ] Implement continuation patterns
- [ ] Add validation

**Day 9: Testing & Refinement**
- [ ] Create reference resolution tests
- [ ] Test edge cases
- [ ] Validate accuracy
- [ ] Performance testing

---

### Phase 4: Dynamic Thresholds (Days 10-11)

**Day 10: Threshold Manager**
- [ ] Create `dynamic_threshold_manager.py`
- [ ] Implement intent-specific thresholds
- [ ] Add context-based adjustments
- [ ] Implement entity-based adjustments

**Day 11: Integration & Testing**
- [ ] Integrate threshold manager
- [ ] Update classification logic
- [ ] Create comprehensive tests
- [ ] Validate improvements

---

### Phase 5: End-to-End Testing (Days 12-14)

**Day 12: Integration Testing**
- [ ] Create multi-turn conversation tests
- [ ] Test reference resolution accuracy
- [ ] Validate context persistence
- [ ] Test threshold adjustments

**Day 13: Performance Testing**
- [ ] Benchmark context operations
- [ ] Test Redis performance
- [ ] Memory usage analysis
- [ ] Latency profiling

**Day 14: Documentation & Deployment**
- [ ] Create API documentation
- [ ] Write user guide
- [ ] Create deployment checklist
- [ ] Update progress tracker

---

## 📊 Success Criteria

### Functional Requirements
- [ ] Context maintained for 5+ turns
- [ ] References resolved with >90% accuracy
- [ ] Follow-up queries handled correctly
- [ ] Context features extracted successfully
- [ ] Confidence boosting improves accuracy by >10%

### Performance Requirements
- [ ] Context retrieval: <5ms
- [ ] Context update: <10ms
- [ ] Reference resolution: <5ms
- [ ] Total overhead: <20ms
- [ ] Redis memory: <100MB per 1000 sessions

### Quality Requirements
- [ ] 100% test coverage
- [ ] All integration tests passing
- [ ] No memory leaks
- [ ] Graceful error handling
- [ ] Comprehensive logging

---

## 🧪 Test Plan

### Unit Tests

**Context Manager Tests:**
- ✓ Session creation and retrieval
- ✓ Context updates
- ✓ Sliding window behavior
- ✓ Entity persistence
- ✓ Context expiration

**Classifier Tests:**
- ✓ Context feature extraction
- ✓ Confidence boosting
- ✓ Reference resolution
- ✓ Intent transition patterns

**Threshold Manager Tests:**
- ✓ Threshold calculation
- ✓ Context-based adjustments
- ✓ Entity-based adjustments

### Integration Tests

**Multi-Turn Conversations:**
```python
Test Scenario 1: Restaurant Follow-up
Turn 1: "restoranlar beyoğlunda" → restaurant_query (0.85)
Turn 2: "orada italyan var mı" → restaurant_query (0.78, boosted from 0.72)
Turn 3: "nasıl giderim" → transport_query (0.82, location from context)

Test Scenario 2: Museum Tour
Turn 1: "sultanahmet müzeleri" → attraction_query (0.88)
Turn 2: "giriş ücreti ne kadar" → attraction_query (0.81, museum from context)
Turn 3: "yakında başka neler var" → attraction_query (0.76, location from context)

Test Scenario 3: Mixed Intents
Turn 1: "topkapı sarayı" → attraction_query (0.91)
Turn 2: "oraya nasıl giderim" → transport_query (0.85, location: Topkapı Palace)
Turn 3: "yakında yemek yeri var mı" → restaurant_query (0.79, location: Sultanahmet)
```

---

## 📁 File Structure

```
backend/services/
├── conversation_context_manager.py     # NEW: Context management
├── context_aware_classifier.py         # NEW: Enhanced classifier
├── dynamic_threshold_manager.py        # NEW: Adaptive thresholds
└── query_preprocessing_pipeline.py     # EXISTING: Updated integration

backend/routes/
├── context_api.py                      # NEW: Context API endpoints

tests/
├── test_context_manager.py            # NEW: Context tests
├── test_context_aware_classifier.py   # NEW: Classifier tests
├── test_multi_turn_conversations.py   # NEW: Integration tests
└── test_reference_resolution.py       # NEW: Resolution tests

docs/
└── WEEK9_10_CONTEXT_AWARE_COMPLETE.md # Completion report
```

---

## 🔍 Example Implementation Snippets

### Context Storage (Redis)
```python
import redis
from datetime import datetime, timedelta

class ConversationContextManager:
    def __init__(self):
        self.redis_client = redis.Redis(
            host='localhost',
            port=6379,
            db=1,
            decode_responses=True
        )
        self.ttl = timedelta(hours=24)  # Context expires after 24h
    
    def get_context(self, session_id: str) -> Optional[ConversationContext]:
        """Retrieve conversation context from Redis"""
        key = f"context:{session_id}"
        data = self.redis_client.get(key)
        
        if not data:
            return None
        
        return ConversationContext.from_json(data)
    
    def update_context(self, session_id: str, turn: Turn):
        """Update conversation context with new turn"""
        context = self.get_context(session_id) or ConversationContext(session_id)
        
        # Add turn to history
        context.add_turn(turn)
        
        # Update persistent entities
        context.update_entities(turn.entities)
        
        # Update intent history
        context.add_intent(turn.intent)
        
        # Store in Redis
        key = f"context:{session_id}"
        self.redis_client.setex(
            key,
            self.ttl,
            context.to_json()
        )
```

### Reference Resolution
```python
def resolve_references(query: str, context: ConversationContext) -> str:
    """Resolve pronouns and implicit references"""
    
    # Detect reference words
    reference_patterns = {
        r'\bit\b': lambda: context.last_mentioned_entity(),
        r'\bthere\b': lambda: context.last_location,
        r'\bthat place\b': lambda: context.last_location,
        r'\bthey\b': lambda: context.last_mentioned_entities(),
    }
    
    resolved = query
    for pattern, resolver in reference_patterns.items():
        if re.search(pattern, query, re.IGNORECASE):
            replacement = resolver()
            if replacement:
                resolved = re.sub(pattern, replacement, resolved, flags=re.IGNORECASE)
    
    return resolved
```

### Confidence Boosting
```python
def boost_confidence_with_context(
    base_confidence: float,
    intent: str,
    context: ConversationContext
) -> float:
    """Boost confidence based on conversation context"""
    
    boost = 0.0
    
    # Boost if same intent as previous turn
    if context.last_intent == intent:
        boost += 0.10
    
    # Boost if intent follows common pattern
    pattern = f"{context.last_intent}->{intent}"
    if pattern in COMMON_INTENT_TRANSITIONS:
        boost += 0.08
    
    # Boost if entities from context present
    if context.has_relevant_entities_for(intent):
        boost += 0.05
    
    # Boost for follow-up queries
    if context.is_follow_up_query():
        boost += 0.07
    
    # Apply boost with ceiling
    boosted = min(base_confidence + boost, 0.99)
    
    return boosted
```

---

## 📈 Expected Improvements

### Classification Accuracy
- **Before:** 81.1% overall accuracy
- **Target After:** 88-92% overall accuracy
- **Follow-up Queries:** 75% → 90%+ accuracy

### User Experience
- **Reference Resolution:** Handle "it", "there", "that" naturally
- **Context Continuity:** Remember last 5 turns
- **Intelligent Suggestions:** Based on conversation history

### Performance
- **Total Added Latency:** <20ms
- **Context Retrieval:** <5ms
- **Reference Resolution:** <5ms
- **Confidence Boosting:** <2ms

---

## 🚀 Getting Started

### Prerequisites
- Redis server running
- Completed Week 7-8 (Preprocessing Pipeline)
- Test environment ready

### Setup Steps
1. Start Redis: `redis-server`
2. Create context manager module
3. Run unit tests
4. Integrate with main backend
5. Run integration tests

---

## 📝 Deliverables

### Code
- [ ] `conversation_context_manager.py` (~400 LOC)
- [ ] `context_aware_classifier.py` (~350 LOC)
- [ ] `dynamic_threshold_manager.py` (~200 LOC)
- [ ] `context_api.py` (~150 LOC)
- [ ] Test files (~600 LOC)

### Documentation
- [ ] API documentation
- [ ] Architecture diagram
- [ ] User guide
- [ ] Completion report (WEEK9_10_CONTEXT_AWARE_COMPLETE.md)

### Tests
- [ ] 15+ unit tests
- [ ] 10+ integration tests
- [ ] 5+ multi-turn conversation tests
- [ ] Performance benchmarks

---

## ⏭️ Next Steps After Completion

### Week 11-12: Testing & Optimization
1. End-to-end integration testing
2. Performance optimization
3. Load testing
4. Production deployment
5. Final documentation

---

**Status:** 📋 PLAN READY  
**Next Action:** Begin Day 1 - Core Infrastructure  
**Target Start:** October 23, 2025  
**Target Completion:** November 6, 2025

---

Let's build context-aware classification! 🚀
