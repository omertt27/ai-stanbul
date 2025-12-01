# URGENT FIX: Query Routing Issue

**Issue**: "Show me the best attractions and landmarks in Istanbul" → Routes to directions handler instead of attractions handler

**Error Message**: "I couldn't identify the locations. Please specify at least a start and end point..."

---

## Root Cause

The query is being misclassified as a **routing/directions intent** when it should be classified as an **attractions information intent**.

**Keywords triggering wrong routing**:
- "show me" → May be triggering location/map intent
- "best attractions" → Should trigger attractions intent, but something is overriding it

---

## Quick Diagnosis

Run this to check how the intent is being detected:

```bash
# Check the backend logs when you send the query
# Look for intent detection output
```

The query contains keywords that should clearly indicate **attractions intent**:
- "best" → recommendations
- "attractions" → POI/attractions
- "landmarks" → historical sites/attractions

But it's being routed to the GPS/directions handler instead.

---

## Likely Causes

### 1. Intent Detection Priority Issue
**File**: `backend/api/chat.py` or intent routing logic

The directions/routing intent may be checked BEFORE the attractions intent, causing:
```python
# WRONG ORDER:
if "show" in query or "me" in query:  # Too broad!
    return route_to_directions()

if "attractions" in query:
    return route_to_attractions()
```

### 2. Overly Broad Routing Keywords
**File**: `backend/services/ai_chat_route_integration.py` lines 1-100

The routing handler might have overly broad keywords like:
- "show" 
- "me"
- "best" (thinking "best route")

### 3. Missing Attractions Keywords
The attractions handler might not be checking for:
- "landmarks"
- "best attractions"
- "show me attractions"

---

## Recommended Fixes

### Fix 1: Add Explicit Attractions Check FIRST

**File**: `backend/api/chat.py` (around intent detection section)

```python
# Check for attractions intent FIRST (before routing)
attractions_keywords = ['attraction', 'attractions', 'landmark', 'landmarks', 
                       'museum', 'mosque', 'palace', 'historical site', 
                       'must see', 'must-see', 'places to visit', 'sights']

# Check if this is clearly an attractions query
query_lower = user_message.lower()
is_attractions = any(keyword in query_lower for keyword in attractions_keywords)

# NOT a routing query if asking for recommendations/info
is_info_request = any(word in query_lower for word in ['best', 'top', 'recommend', 'what are', 'tell me about', 'show me the'])
is_not_routing = not any(word in query_lower for word in ['from', 'to', 'route', 'directions', 'how to get', 'take me'])

if is_attractions and is_info_request and is_not_routing:
    # Route to attractions handler, NOT directions
    # ... handle attractions query
```

### Fix 2: Update Routing Intent Detection

**File**: `backend/services/ai_chat_route_integration.py`

Make the routing detection MORE SPECIFIC:

```python
def should_handle_routing(self, message: str) -> bool:
    """Determine if this is a routing/directions query"""
    message_lower = message.lower()
    
    # MUST have explicit routing keywords
    routing_keywords = [
        'route from', 'directions to', 'how to get to', 'how do i get',
        'take me to', 'navigate to', 'from X to Y', 'way to'
    ]
    
    has_routing_intent = any(kw in message_lower for kw in routing_keywords)
    
    # EXCLUDE if it's clearly an information request
    info_keywords = ['what are', 'show me the', 'tell me about', 'recommend', 'best', 'top']
    is_info_request = any(kw in message_lower for kw in info_keywords)
    
    # EXCLUDE if asking about attractions/POIs (not directions)
    attractions_keywords = ['attractions', 'landmarks', 'museums', 'places to visit']
    is_about_attractions = any(kw in message_lower for kw in attractions_keywords)
    
    # Only handle if clear routing intent AND not info request
    return has_routing_intent and not (is_info_request and is_about_attractions)
```

### Fix 3: Improve Intent Classification Order

**File**: `backend/api/chat.py` (main chat endpoint)

Ensure intent checks happen in this order:

```python
# 1. INFORMATION REQUESTS (attractions, restaurants, etc.)
if is_attractions_info_request(message):
    return handle_attractions()

if is_restaurant_info_request(message):
    return handle_restaurants()

# 2. ROUTING/DIRECTIONS (only if NOT info request)
if is_routing_request(message):
    return handle_routing()

# 3. GENERAL (fallback)
return handle_general()
```

---

## Testing After Fix

Test these queries to ensure correct routing:

### Should Route to ATTRACTIONS Handler:
✅ "Show me the best attractions in Istanbul"
✅ "What are the top landmarks?"
✅ "Best places to visit in Istanbul"
✅ "Recommend some attractions"
✅ "Tell me about historical sites"

### Should Route to DIRECTIONS Handler:
✅ "How do I get from Sultanahmet to Taksim?"
✅ "Route from my location to Hagia Sophia"
✅ "Directions to Topkapi Palace"
✅ "Take me to Galata Tower"

### Should Route to RESTAURANTS Handler:
✅ "Best restaurants in Sultanahmet"
✅ "Recommend a place to eat"
✅ "Where can I find good kebab?"

---

## Quick Implementation

1. Find the intent detection logic in `backend/api/chat.py`
2. Add attractions keyword check BEFORE routing check
3. Make routing detection more specific (require "from X to Y" or "directions to")
4. Test with the problematic query
5. Verify it now routes to attractions handler

---

## Alternative: Check Intent Detector

If you're using a separate intent detector service:

**File**: Look for `intent_detector.py`, `intent_classifier.py`, or similar

The model/rules may need retraining or rule updates to:
- Prioritize "attractions" over "routing" for ambiguous queries
- Recognize "show me attractions" as info request, not directions request

---

## Expected Outcome After Fix

**Query**: "Show me the best attractions and landmarks in Istanbul"

**Current (Wrong)**: Routes to directions → "I couldn't identify locations..."

**After Fix (Correct)**: Routes to attractions → Lists Hagia Sophia, Topkapi Palace, Blue Mosque, etc. with details

---

**Status**: Urgent fix needed in intent routing logic
**Priority**: High (affects basic attraction queries)
**Estimated Fix Time**: 15-30 minutes
