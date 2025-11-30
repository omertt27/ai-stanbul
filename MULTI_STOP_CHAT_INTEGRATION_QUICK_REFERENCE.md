# Multi-Stop Chat Integration - Quick Reference

## 🎯 What This Does

Users can now ask for multi-stop itineraries in natural language:
- ✅ "Plan a tour of Hagia Sophia, Blue Mosque, and Grand Bazaar"
- ✅ "Visit Topkapi Palace, then Basilica Cistern, then Spice Bazaar"
- ✅ "Create an accessible itinerary for Galata Tower, Istiklal Street, and Taksim"

System automatically:
1. Detects multi-stop vs single route queries
2. Extracts location names from natural language
3. Plans optimized itinerary
4. Formats beautiful response with timeline

---

## 🚀 Quick Start

### Basic Usage
```python
from backend.services.ai_chat_route_integration import get_chat_route_handler

handler = get_chat_route_handler()
result = handler.handle_route_request("Visit Hagia Sophia, Blue Mosque, Grand Bazaar")

print(result['message'])  # User-friendly response
route_data = result['route_data']  # Data for map
```

### With User Context
```python
user_context = {
    'accessible_mode': True,
    'budget': 'low',
    'current_location': (41.0086, 28.9802)
}

result = handler.handle_route_request(
    "Plan my day visiting 3 museums",
    user_context=user_context
)
```

---

## 📝 Supported Query Types

### Multi-Stop Queries (NEW!)
- "Plan a tour of A, B, and C"
- "Visit A, B, C today"
- "Create an itinerary for A, then B, then C"
- "Best route to see A, B, C"
- "I want to visit A and B and C"

### Single Route Queries (Existing)
- "Route from A to B"
- "How do I get to A?"
- "Directions from A to B"

### Detection Logic
```python
# Multi-stop if:
- 3+ locations mentioned
- Keywords: 'itinerary', 'plan', 'tour', 'visit multiple'
- List pattern: "A, B, and C"
- Connectors: 'then', 'also', 'and then'

# Single route if:
- "from X to Y" pattern
- Only 1-2 locations
- Direction keywords: 'route', 'directions'
```

---

## 🎨 Response Format

### Multi-Stop Response
```python
{
    'type': 'multi_stop_itinerary',
    'message': '🗺️ Multi-Stop Itinerary Planned!...',  # Formatted text
    'route_data': {
        'type': 'multi_stop_itinerary',
        'stops': [
            {
                'name': 'Hagia Sophia',
                'coordinates': (41.0086, 28.9802),
                'category': 'museum',
                'duration': 90,
                'accessibility': 'partial'
            },
            ...
        ],
        'segments': [
            {
                'from': 'Hagia Sophia',
                'to': 'Blue Mosque',
                'distance_km': 0.5,
                'duration_min': 10,
                'modes': ['walking'],
                'cost_tl': 0.0
            },
            ...
        ],
        'summary': {
            'total_stops': 3,
            'total_distance_km': 2.5,
            'total_travel_time_min': 30,
            'total_visit_time_min': 225,
            'total_time_min': 255,
            'total_cost_tl': 15.50,
            'strategy': 'shortest_time',
            'accessibility_friendly': True
        },
        'timeline': [
            {'time': '09:00', 'type': 'arrival', 'location': 'Hagia Sophia'},
            {'time': '09:00 - 10:30', 'type': 'visit', 'duration': 90},
            {'time': '10:30 - 10:40', 'type': 'travel', 'modes': ['walking']},
            ...
        ]
    }
}
```

### Error Response
```python
{
    'type': 'error',
    'message': 'I need at least 2 locations to plan an itinerary...'
}
```

### Not a Route Query
```python
None  # Returns None if not recognized as route request
```

---

## 🔧 Key Methods

### 1. Signal Detection
```python
handler._is_multi_stop_request(message: str) -> bool
```
Determines if query is multi-stop or single route.

### 2. POI Extraction
```python
handler._extract_poi_names(message: str) -> List[str]
```
Extracts location names from natural language.

### 3. Multi-Stop Handler
```python
handler._handle_multi_stop_request(
    message: str,
    user_context: Optional[Dict] = None
) -> Dict
```
Plans itinerary and formats response.

### 4. Response Formatter
```python
handler._format_multi_stop_response(
    itinerary: MultiStopItinerary,
    original_message: str
) -> Dict
```
Creates user-friendly response with map data.

---

## 🗺️ POI Database

Currently supported locations (12 major attractions):

**Sultanahmet Area:**
- Hagia Sophia
- Blue Mosque
- Topkapi Palace
- Basilica Cistern

**Beyoğlu Area:**
- Galata Tower
- Istiklal Street
- Taksim Square

**Bosphorus:**
- Dolmabahçe Palace
- Maiden Tower

**Shopping:**
- Grand Bazaar
- Spice Bazaar

**Asian Side:**
- Kadıköy

*Want more? Expand POI database in `multi_stop_route_planner.py`*

---

## ⚙️ Optimization Strategies

User query determines strategy:

| Keyword | Strategy | Effect |
|---------|----------|--------|
| "accessible" | ACCESSIBLE_FIRST | Prioritizes wheelchair-accessible routes |
| "distance" | SHORTEST_TOTAL_DISTANCE | Minimizes walking distance |
| "nearest" | NEAREST_NEIGHBOR | Greedy nearest-neighbor order |
| (default) | SHORTEST_TOTAL_TIME | Minimizes total travel time |

**Example:**
```python
"Create an accessible itinerary for A, B, C"
→ Uses ACCESSIBLE_FIRST strategy
```

---

## 🧪 Testing

### Run All Tests
```bash
python test_chat_multi_stop_integration.py
```

### Current Test Results
- ✅ POI Extraction: 100% (4/4)
- ✅ Response Formatting: 100% (5/5)
- ✅ Error Handling: 100% (3/3)
- ⚠️ Signal Detection: 62.5% (5/8)
- ⚠️ Multi-Stop Planning: Routing data incomplete in tests
- ⚠️ Real-World Queries: 66.7% (2/3)

### Test Individual Components
```python
from test_chat_multi_stop_integration import test_poi_extraction
test_poi_extraction()
```

---

## 🐛 Common Issues

### Issue: "Multi-stop planner not available"
**Cause:** Import error  
**Fix:** Check routing infrastructure imports in `multi_stop_route_planner.py`

### Issue: Distance/duration = 0
**Cause:** Routing engine graph not initialized  
**Fix:** Ensure routing engine fully loaded before planning

### Issue: POI not recognized
**Cause:** Location not in database  
**Fix:** Add to `KNOWN_LOCATIONS` in `ai_chat_route_integration.py` or POI database in `multi_stop_route_planner.py`

### Issue: Wrong query type detected
**Cause:** Ambiguous phrasing  
**Fix:** Refine `_is_multi_stop_request()` patterns

---

## 📦 Dependencies

- `multi_stop_route_planner.py` - Core itinerary planning
- `graph_routing_engine.py` - Route finding between stops
- `route_optimizer.py` - Optimization strategies
- `transportation_directions_service.py` - Transit routing

---

## 🎯 Integration Points

### Frontend (Map Display)
```javascript
// Use route_data.stops for markers
response.route_data.stops.forEach(stop => {
    addMarker(stop.coordinates, stop.name, stop.category);
});

// Use route_data.segments for polylines
response.route_data.segments.forEach(segment => {
    drawRoute(segment.from, segment.to, segment.modes);
});
```

### Backend (Chat Handler)
```python
def handle_user_message(message, user_id):
    # Try route handler first
    route_response = process_chat_route_request(message)
    
    if route_response:
        # Log analytics
        log_route_request(user_id, route_response['type'])
        return route_response
    
    # Fall back to general chat
    return general_ai_response(message)
```

---

## 📊 Analytics to Track

1. **Query Type Distribution**
   - % single route vs multi-stop
   - Most common #stops requested

2. **POI Popularity**
   - Which locations mentioned most
   - Common combinations

3. **Success Rates**
   - % queries successfully planned
   - Common failure reasons

4. **Performance**
   - Response time by #stops
   - Routing engine latency

5. **User Satisfaction**
   - Follow-up queries
   - Itinerary completion rate

---

## 🚀 Deployment Checklist

- [ ] Test in production environment
- [ ] Verify routing engine initialized
- [ ] Load test with 100 requests
- [ ] Monitor error rates
- [ ] A/B test with 10% users
- [ ] Update API docs
- [ ] Train support team on new feature

---

## 📚 Related Documentation

- `MULTI_STOP_CHAT_INTEGRATION_COMPLETE.md` - Full implementation details
- `ACCESSIBILITY_EXPANSION_COMPLETE.md` - Accessibility features
- `WHATS_NEXT_IMPLEMENTATION_ROADMAP.md` - Future enhancements
- `multi_stop_route_planner.py` - Core planner documentation

---

## 💡 Tips & Best Practices

1. **Always validate POI extraction** - Log unrecognized locations
2. **Monitor routing failures** - Alert if success rate < 90%
3. **Cache common itineraries** - Speed up popular routes
4. **Expand POI database gradually** - Start with most requested
5. **A/B test signal detection** - Improve accuracy over time

---

**Questions?** Check the full documentation or run the test suite!

**Author:** Istanbul AI Team  
**Last Updated:** November 30, 2025
