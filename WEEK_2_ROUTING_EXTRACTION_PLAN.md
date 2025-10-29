# Week 2: Routing Layer Extraction Plan

## Objective
Extract routing logic from `main_system.py` into a dedicated routing layer, further improving modularity and maintainability.

## Current State (After Week 1)
- ✅ Service initialization modularized (ServiceInitializer)
- ✅ ML handler initialization modularized (HandlerInitializer)
- ✅ System configuration centralized (SystemConfig)
- 📊 Current line count: 2,891 lines

## Week 2 Target
- 🎯 Extract routing logic to dedicated modules
- 🎯 Reduce main_system.py by additional 400-500 lines
- 🎯 Target line count: ~2,400-2,500 lines

## Architecture Design

### 1. Intent Classifier (`routing/intent_classifier.py`)
**Purpose**: Classify user intent from message

**Responsibilities**:
- Parse user message
- Detect intent keywords
- Return classified intent (restaurant, attraction, transportation, etc.)
- Handle multi-intent queries

**Methods**:
```python
class IntentClassifier:
    def classify_intent(message: str, entities: Dict, context: ConversationContext) -> str
    def detect_multiple_intents(message: str, entities: Dict) -> List[str]
    def is_daily_talk_query(message: str) -> bool
```

**Code to Extract**:
- `_classify_intent_with_context()` (lines ~900-1050)
- `_detect_multiple_intents()` (lines ~1850-1875)
- `_is_daily_talk_query()` (lines ~700-750)

---

### 2. Entity Extractor (`routing/entity_extractor.py`)
**Purpose**: Extract entities from user message

**Responsibilities**:
- Use IstanbulEntityRecognizer
- Extract locations, cuisines, districts, etc.
- Parse temporal expressions
- Extract budget/price indicators

**Methods**:
```python
class EntityExtractor:
    def extract_entities(message: str, context: ConversationContext) -> Dict[str, Any]
    def extract_location(message: str) -> Optional[str]
    def extract_budget(message: str) -> Optional[str]
    def extract_temporal(message: str) -> Optional[str]
```

**Integration**:
- Wraps existing `IstanbulEntityRecognizer`
- Adds contextual enhancements
- Provides unified entity extraction interface

---

### 3. Response Router (`routing/response_router.py`)
**Purpose**: Route queries to appropriate handlers

**Responsibilities**:
- Route based on intent
- Select appropriate handler (ML vs standard)
- Handle fallbacks
- Manage handler priorities

**Methods**:
```python
class ResponseRouter:
    def route_query(
        message: str,
        intent: str,
        entities: Dict,
        user_profile: UserProfile,
        context: ConversationContext,
        handlers: Dict[str, Any]
    ) -> Union[str, Dict[str, Any]]
    
    def _route_to_ml_handler(intent: str, ...) -> Optional[str]
    def _route_to_standard_handler(intent: str, ...) -> str
```

**Code to Extract**:
- `_generate_contextual_response()` routing logic (lines ~1050-1400)
- Handler selection logic
- Fallback management

---

### 4. Query Preprocessor (`routing/query_preprocessor.py`)
**Purpose**: Preprocess and enhance queries before routing

**Responsibilities**:
- Normalize text
- Detect language (Turkish/English)
- Extract neural insights (if available)
- Build query context

**Methods**:
```python
class QueryPreprocessor:
    def preprocess_query(
        message: str,
        user_id: str,
        user_profile: UserProfile,
        neural_processor: Optional[Any]
    ) -> Dict[str, Any]
    
    def detect_language(message: str) -> str
    def extract_neural_insights(message: str, processor: Any) -> Dict
```

---

## Implementation Steps

### Step 1: Create Routing Module Structure
```bash
mkdir -p istanbul_ai/routing
touch istanbul_ai/routing/__init__.py
touch istanbul_ai/routing/intent_classifier.py
touch istanbul_ai/routing/entity_extractor.py
touch istanbul_ai/routing/response_router.py
touch istanbul_ai/routing/query_preprocessor.py
```

### Step 2: Implement Intent Classifier
1. Create `IntentClassifier` class
2. Extract intent classification logic from main_system.py
3. Add comprehensive intent keywords
4. Implement multi-intent detection
5. Add unit tests

### Step 3: Implement Entity Extractor
1. Create `EntityExtractor` class
2. Wrap existing `IstanbulEntityRecognizer`
3. Add contextual entity extraction
4. Add budget/temporal extraction
5. Add unit tests

### Step 4: Implement Response Router
1. Create `ResponseRouter` class
2. Extract routing logic from `_generate_contextual_response()`
3. Implement ML handler routing
4. Implement standard handler routing
5. Add fallback handling
6. Add unit tests

### Step 5: Implement Query Preprocessor
1. Create `QueryPreprocessor` class
2. Add text normalization
3. Add language detection
4. Add neural insights extraction
5. Add unit tests

### Step 6: Integrate Routing Layer
1. Update main_system.py to use routing layer
2. Replace `_classify_intent_with_context()` with `IntentClassifier`
3. Replace `_generate_contextual_response()` routing with `ResponseRouter`
4. Add `QueryPreprocessor` to `process_message()`
5. Run integration tests

### Step 7: Testing & Validation
1. Run existing test suite
2. Add routing-specific tests
3. Test all intent types
4. Test multi-intent queries
5. Test ML handler fallbacks
6. Performance testing

---

## Success Criteria

### Functional
- ✅ All intents correctly classified
- ✅ All entities correctly extracted
- ✅ Queries correctly routed to handlers
- ✅ ML handlers take priority when available
- ✅ Graceful fallbacks when handlers fail
- ✅ Multi-intent queries handled

### Code Quality
- ✅ main_system.py reduced by 400-500 lines
- ✅ Routing logic in dedicated modules
- ✅ Single responsibility principle applied
- ✅ Comprehensive unit tests (>80% coverage)
- ✅ Clear documentation

### Performance
- ✅ No performance degradation
- ✅ Intent classification < 50ms
- ✅ Entity extraction < 100ms
- ✅ Routing decision < 20ms

---

## Expected File Structure (After Week 2)

```
istanbul_ai/
├── main_system.py (~2,400 lines, -500 from Week 2)
├── initialization/
│   ├── __init__.py
│   ├── service_initializer.py
│   ├── handler_initializer.py
│   └── system_config.py
├── routing/  [NEW]
│   ├── __init__.py
│   ├── intent_classifier.py (~200 lines)
│   ├── entity_extractor.py (~150 lines)
│   ├── response_router.py (~250 lines)
│   └── query_preprocessor.py (~100 lines)
├── core/
│   ├── models.py
│   ├── entity_recognition.py
│   ├── response_generator.py
│   └── user_management.py
└── services/
    └── ...
```

---

## Timeline

- **Day 1**: Create routing module structure + IntentClassifier
- **Day 2**: EntityExtractor + QueryPreprocessor
- **Day 3**: ResponseRouter (main routing logic)
- **Day 4**: Integration into main_system.py
- **Day 5**: Testing, validation, documentation

---

## Risk Mitigation

### Risk 1: Breaking existing functionality
**Mitigation**: 
- Comprehensive unit tests before refactoring
- Integration tests after each change
- Keep old code commented during transition

### Risk 2: Performance degradation
**Mitigation**:
- Benchmark before and after
- Profile routing layer
- Optimize hot paths

### Risk 3: Complex dependencies
**Mitigation**:
- Use dependency injection
- Clear interfaces between components
- Minimize coupling

---

## Next Action
Start with **Step 1: Create Routing Module Structure** and **Step 2: Implement Intent Classifier**

Ready to proceed? 🚀
