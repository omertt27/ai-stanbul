# Route Planning GPS Fix

## Issue
When users provide both origin and destination in their query (e.g., "how can I go to Taksim from Kadıköy"), the system:
1. ✅ Correctly extracts both locations
2. ✅ Generates the route/map
3. ❌ BUT still asks users to enable GPS in the text response

## Root Cause
The LLM system prompt instructs: "Start location: Explicitly mentioned origin OR user's GPS location (if no origin mentioned) OR ask user"

This causes the LLM to ask for GPS even when:
- Both origin and destination are extracted from the query
- The map service already generated a route
- No GPS is actually needed

## Solution
Add context to the prompt builder to inform the LLM when both locations have been extracted, so it doesn't ask for GPS unnecessarily.

### Implementation Steps

1. **Update map_visualization_service.py**: Return extracted location information
   - Add `has_origin` and `has_destination` flags to map_data
   - This tells downstream components that locations were found

2. **Update context.py**: Pass location extraction info to the response
   - Include location extraction status in the context
   - This informs the prompt builder

3. **Update prompts.py**: Add conditional instructions based on extracted locations
   - If both origin and destination are extracted → Don't ask for GPS
   - If only destination is extracted → Can use GPS or ask for origin
   - If neither is extracted → Provide general help

## Files to Edit
- `backend/services/map_visualization_service.py` - Add location extraction flags
- `backend/services/llm/context.py` - Pass extraction info to prompt
- `backend/services/llm/prompts.py` - Update prompt logic to use extraction info

## Expected Behavior After Fix
**Query**: "how can I go to Taksim from Kadıköy"
- System extracts: origin="Kadıköy", destination="Taksim"
- System generates route with both locations
- LLM responds with route details WITHOUT asking for GPS
- Map shows the complete route

**Query**: "how can I go to Taksim"
- System extracts: origin=None, destination="Taksim"
- If user has GPS: Use GPS as origin, generate route
- If no GPS: Ask user for starting location
- Map shows destination marker (or route if GPS available)
