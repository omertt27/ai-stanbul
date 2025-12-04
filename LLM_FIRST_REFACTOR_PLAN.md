# LLM-First Architecture Refactor Plan

## Current State Analysis

### Pattern Matching Still Used In:

1. **`chat.py` Line 541-563**: Fallback `is_information_request()` function
   - Uses regex-based info/direction keyword detection
   - Only triggers when LLM confidence < 0.7
   - Should be replaced with LLM fallback or removed

2. **`chat.py` Line 1099-1101**: Transportation keyword detection
   - Used in error fallback when Pure LLM unavailable
   - Should use lightweight LLM fallback instead

3. **`chat.py` Line 1245-1252**: `_is_hidden_gems_request()` function
   - Regex-based hidden gems detection
   - Should be handled by LLM intent classifier

4. **Hidden Gems GPS Integration**: Pattern-based detection
   - Should rely on LLM intent classification
   - Patterns should only be emergency fallback

### LLM-First Architecture Achieved:

✅ **Multi-Intent Handler**: Pure LLM-based intent detection
✅ **Location Resolver**: LLM-powered location extraction
✅ **Route Handler**: Uses LLM intent for routing decisions
✅ **Context Resolution**: LLM-based reference resolution
✅ **Response Synthesis**: LLM-powered response generation
✅ **Smart Routing**: LLM intent directly routes to handlers (Phase 2.5)

---

## Refactor Strategy

### Phase 1: Remove Pattern-Based Intent Detection (Priority: HIGH)

**Goal**: Make LLM the single source of truth for intent classification

**Actions**:
1. Remove `is_information_request()` fallback function
2. Trust LLM intent even with confidence 0.5+ (currently 0.7)
3. If LLM fails completely, use simple direct fallback (not pattern matching)

**Files to Edit**:
- `backend/api/chat.py`: Remove lines 541-575 (fallback regex detection)
- Replace with trust in LLM or generic fallback

### Phase 2: Eliminate Hidden Gems Pattern Matching (Priority: HIGH)

**Goal**: Let LLM classify hidden gems intent

**Actions**:
1. Remove `_is_hidden_gems_request()` function
2. Add "hidden_gems" as a primary intent in Multi-Intent Handler
3. Hidden gems handler should only activate if LLM says so

**Files to Edit**:
- `backend/api/chat.py`: Remove `_is_hidden_gems_request()`
- `backend/services/llm/multi_intent_handler.py`: Ensure hidden_gems is a first-class intent
- `backend/services/hidden_gems_gps_integration.py`: Remove pattern detection logic

### Phase 3: Improve LLM Fallback Strategy (Priority: MEDIUM)

**Goal**: When LLM unavailable, use lightweight alternatives (not regex)

**Actions**:
1. Create a minimal "emergency fallback" that gives helpful generic responses
2. Remove transportation_keywords in fallback (lines 1099-1101)
3. Show clear error message that LLM is down + generic suggestions

**Files to Edit**:
- `backend/api/chat.py`: Simplify lines 1090-1130 (fallback handler)

### Phase 4: Confidence Threshold Tuning (Priority: MEDIUM)

**Goal**: Lower confidence threshold to trust LLM more

**Actions**:
1. Reduce confidence threshold from 0.7 to 0.5 for intent routing
2. Add logging for confidence ranges to monitor quality
3. Let LLM handle more queries, even if slightly uncertain

**Files to Edit**:
- `backend/api/chat.py`: Line ~580 (confidence check)

### Phase 5: Frontend Map Rendering (Priority: HIGH)

**Goal**: Display map visualizations from `map_data` field

**Actions**:
1. Update `Chatbot.jsx` to detect `map_data` in responses
2. Render interactive map using Leaflet/MapLibre
3. Show route visualization, markers, and polylines

**Files to Edit**:
- `frontend/src/Chatbot.jsx`: Add map rendering component
- `frontend/src/components/MapVisualization.jsx`: Create new component (if needed)

---

## Implementation Order

1. ✅ **Phase 2.5 Complete**: LLM-based smart routing for route queries
2. **Next: Phase 1** - Remove fallback pattern detection for info requests
3. **Next: Phase 2** - Eliminate hidden gems patterns
4. **Next: Phase 4** - Lower confidence threshold to 0.5
5. **Next: Phase 5** - Frontend map rendering
6. **Next: Phase 3** - Improve emergency fallback

---

## Expected Outcomes

### User Experience:
- More accurate intent detection
- Better handling of complex/ambiguous queries
- Map visualizations in chat responses
- Fewer "I don't understand" messages

### Architecture:
- Single source of truth (LLM) for intent
- Cleaner code without regex sprawl
- Easier to extend with new intents
- Better observability (LLM confidence logs)

### Performance:
- Slightly slower (more LLM calls) but more accurate
- Can add caching for common queries
- Circuit breaker prevents cascading failures

---

## Testing Plan

1. Test route queries with various phrasings
2. Test hidden gems discovery
3. Test restaurant/attraction searches
4. Test with GPS enabled/disabled
5. Test LLM unavailable scenario
6. Test ambiguous queries (should ask for clarification)
7. Test frontend map rendering

---

## Rollback Plan

If LLM-first causes issues:
1. Keep pattern matching as "backup classifier"
2. Use patterns only if LLM confidence < 0.3
3. Add feature flag to toggle LLM-first mode

---

## Next Steps

Starting with **Phase 1**: Remove fallback pattern detection
