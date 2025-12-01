# Fix: Map Not Showing for Neighborhood/Attraction Queries

**Date**: December 1, 2025  
**Status**: Issue Identified - Fix Required  
**Priority**: HIGH  

---

## 🐛 Problem Description

When users ask about neighborhoods or attractions (e.g., "tell me about Beşiktaş district"), the LLM response **says** it will show a map:

```
🗺️ Haritada göstereceğim/I'll show you this route on a map below. ⬇️
```

But **NO MAP ACTUALLY APPEARS** because:
1. The `needs_map` signal is not detected (no explicit "map" keyword in query)
2. The context builder only generates `map_data` when `needs_map` or `needs_gps_routing` signals are True
3. The response returns `map_data: null` to the frontend

---

## 🔍 Root Cause Analysis

### Current Logic (backend/services/llm/context.py, line 210):

```python
# Generate map visualization
if (signals.get('needs_map') or signals.get('needs_gps_routing')) and self.map_service:
    try:
        context['map_data'] = await self._generate_map(
            query=query,
            signals=signals,
            user_location=user_location,
            language=language
        )
```

**Problem**: This ONLY generates maps when:
- User explicitly asks for a "map" (needs_map signal)
- User asks for directions/routing (needs_gps_routing signal)

**Missing**: Maps are NOT generated for:
- Neighborhood queries ("tell me about Beşiktaş")
- Attraction queries ("what can I see in Sultanahmet?")
- Restaurant recommendations ("best restaurants in Kadıköy")

---

## ✅ Solution

### Option 1: Expand `needs_map` Signal Detection (RECOMMENDED)

Add neighborhood and attraction keywords to the `needs_map` signal patterns in `backend/services/llm/signals.py`:

```python
'needs_map': {
    'en': [
        r'\b(map|show.*map|visual|locate|location)\b',
        r'\b(where.*is|where.*are|find.*on.*map)\b',
        # ADD THESE:
        r'\b(neighborhood|district|area|quarter)\b',
        r'\b(attractions?|landmarks?|sights?|places\s+to\s+visit)\b',
        r'\b(restaurants?.*in|cafes?.*in|eat.*in)\b'
    ],
    'tr': [
        r'\b(harita|haritada\s+göster|konum|yer)\b',
        r'\b(nerede|haritada\s+bul)\b',
        # ADD THESE:
        r'\b(mahalle|semt|bölge|çevre)\b',
        r'\b(gezilecek\s+yer|görülecek\s+yer|turistik\s+yer)\b',
        r'\b(restoranlar|kafeler|yemek)\b'
    ]
}
```

### Option 2: Auto-Generate Maps for Certain Intents (BETTER)

Modify `backend/services/llm/context.py` to automatically generate maps when specific signals are detected:

```python
# Generate map visualization
should_generate_map = (
    signals.get('needs_map') or 
    signals.get('needs_gps_routing') or
    signals.get('needs_neighborhood') or  # NEW
    signals.get('needs_attraction') or     # NEW
    signals.get('needs_restaurant')        # NEW
)

if should_generate_map and self.map_service:
    try:
        context['map_data'] = await self._generate_map(
            query=query,
            signals=signals,
            user_location=user_location,
            language=language
        )
    except Exception as e:
        logger.warning(f"Map generation failed: {e}")
```

### Option 3: Always Generate Maps (SIMPLEST)

Always generate map data when `map_service` is available and the query is location-related:

```python
# Generate map visualization for all location-based queries
if self.map_service and (user_location or any([
    signals.get('needs_restaurant'),
    signals.get('needs_attraction'),
    signals.get('needs_neighborhood'),
    signals.get('needs_transportation'),
    signals.get('needs_hidden_gems'),
    signals.get('needs_map'),
    signals.get('needs_gps_routing')
])):
    try:
        context['map_data'] = await self._generate_map(
            query=query,
            signals=signals,
            user_location=user_location,
            language=language
        )
    except Exception as e:
        logger.warning(f"Map generation failed: {e}")
```

---

## 🛠️ Implementation Steps

### Step 1: Choose Solution (RECOMMENDED: Option 2)

Option 2 is best because:
- ✅ Generates maps when contextually relevant
- ✅ Doesn't generate unnecessary maps for general queries
- ✅ Minimal code changes
- ✅ Easy to extend with more signals

### Step 2: Update context.py

**File**: `backend/services/llm/context.py`  
**Line**: ~210

**Change from**:
```python
if (signals.get('needs_map') or signals.get('needs_gps_routing')) and self.map_service:
```

**Change to**:
```python
should_generate_map = (
    signals.get('needs_map') or 
    signals.get('needs_gps_routing') or
    signals.get('needs_neighborhood') or
    signals.get('needs_attraction') or
    signals.get('needs_restaurant')
)

if should_generate_map and self.map_service:
```

### Step 3: Remove LLM's False Map Promise

**File**: `IMPROVED_PROMPT_TEMPLATES.py`  
**Location**: Neighborhood intent prompt

**Remove this line from LLM prompts**:
```
🗺️ Haritada göstereceğim/I'll show you this route on a map below. ⬇️
```

The LLM shouldn't promise a map in the text - the map should just appear automatically when relevant.

### Step 4: Test

Test these queries and verify map appears:
1. "tell me about Beşiktaş district" ✓ Should show map
2. "best restaurants in Sultanahmet" ✓ Should show map  
3. "what attractions are in Kadıköy?" ✓ Should show map
4. "what time does Hagia Sophia open?" ✓ Should show map
5. "what is the weather in Istanbul?" ✗ Should NOT show map

---

## 📊 Expected Behavior After Fix

### Before (Current):
```json
{
  "response": "Beşiktaş is beautiful... 🗺️ I'll show you on a map below",
  "map_data": null  ❌
}
```

### After (Fixed):
```json
{
  "response": "Beşiktaş is beautiful... Here are must-see spots:",
  "map_data": {  ✅
    "type": "neighborhood",
    "center": {"lat": 41.0433, "lon": 29.0070},
    "markers": [
      {"name": "Dolmabahçe Palace", "lat": 41.0391, "lon": 29.0008},
      {"name": "Yıldız Park", "lat": 41.0486, "lon": 29.0119},
      ...
    ],
    "zoom": 14
  }
}
```

---

## 🚀 Deployment

1. **Development**: Test fix locally
2. **Staging**: Deploy to staging environment
3. **Production**: Deploy after testing confirms maps appear
4. **Monitor**: Check that map_data is being generated in logs

---

## 📝 Related Files

- `backend/services/llm/context.py` (line ~210) - Main fix location
- `backend/services/llm/signals.py` - Signal patterns
- `IMPROVED_PROMPT_TEMPLATES.py` - Prompt templates (remove false promises)
- `backend/api/chat.py` - API endpoint (no changes needed)

---

## ⚠️ Important Notes

1. **Don't promise maps in prompts**: The LLM shouldn't say "I'll show you on a map" unless the system actually generates map_data.

2. **Map service must be available**: Ensure `map_service` is initialized in the context builder.

3. **User location optional**: Maps should work even without user GPS location (use default Istanbul center).

4. **Performance**: Generating maps adds ~100-200ms latency - acceptable for better UX.

---

**NEXT ACTION**: Implement Option 2 (Auto-generate maps for certain intents) in `backend/services/llm/context.py`.
