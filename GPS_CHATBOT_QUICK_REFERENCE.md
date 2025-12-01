# 🚀 GPS Chatbot Integration - Developer Quick Reference

## 1-Minute Overview

When user types **"How can I go to Taksim?"**:
1. Frontend sends message + GPS location to `/api/chat/pure-llm`
2. Backend detects route request + extracts destination
3. Uses GPS as start point (or requests permission if not available)
4. Plans route with OSRM
5. Returns chat response + map data
6. Frontend displays message + map with route
7. User can start turn-by-turn navigation

## Key Files

| Component | File | Purpose |
|-----------|------|---------|
| Backend Core | `backend/services/ai_chat_route_integration.py` | Route request handling + GPS extraction |
| Chat API | `backend/api/chat.py` | Chat endpoint with GPS support |
| GPS Engine | `backend/services/gps_turn_by_turn_navigation.py` | Turn-by-turn navigation |
| Frontend Map | `frontend/gps_navigation_map.html` | Map component |
| Frontend Chat | `frontend/gps_chat_integration.html` | Chat + map UI |

## Backend: Handle Route Request with GPS

```python
from services.ai_chat_route_integration import get_chat_route_handler

handler = get_chat_route_handler()

# User context with GPS
user_context = {
    'gps': {'lat': 41.0086, 'lon': 28.9802},  # From frontend
    'preferences': {'language': 'en'}
}

# Handle request
result = handler.handle_route_request(
    message="How can I go to Taksim?",
    user_context=user_context
)

# Result contains:
# - result['message']: Chat response text
# - result['map_data']: Route coordinates, start, end
# - result['suggestions']: Next actions
# - result['type']: 'route' or 'gps_permission_required'
```

## Frontend: Send Message with GPS

```javascript
async function sendMessage(text) {
    // Get GPS location
    const location = await getCurrentLocation();
    
    // Send to backend
    const response = await fetch('/api/chat/pure-llm', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            message: text,
            user_location: location,  // {lat: 41.0086, lon: 28.9802}
            session_id: sessionId
        })
    });
    
    const data = await response.json();
    
    // Display response
    displayMessage(data.response);
    
    // Show route on map if available
    if (data.map_data) {
        displayRoute(data.map_data);
    }
}

function getCurrentLocation() {
    return new Promise((resolve, reject) => {
        if (!navigator.geolocation) {
            reject('GPS not supported');
            return;
        }
        
        navigator.geolocation.getCurrentPosition(
            position => resolve({
                lat: position.coords.latitude,
                lon: position.coords.longitude
            }),
            error => reject(error)
        );
    });
}
```

## API Request/Response Examples

### Example 1: Route Request with GPS

**Request:**
```json
POST /api/chat/pure-llm
{
  "message": "How can I go to Taksim?",
  "user_location": {
    "lat": 41.0086,
    "lon": 28.9802
  },
  "session_id": "user_123"
}
```

**Response:**
```json
{
  "response": "🗺️ Here's your walking route!\n\n📍 From: Hagia Sophia\n📍 To: Taksim Square\n📏 Distance: 6.59 km\n⏱️ Time: 12 minutes",
  "session_id": "user_123",
  "intent": "route_planning",
  "confidence": 1.0,
  "map_data": {
    "start": [41.0086, 28.9802],
    "end": [41.0370, 28.9850],
    "route": [[41.0086, 28.9802], [41.0090, 28.9805], ...],
    "distance_km": 6.59,
    "duration_min": 12
  },
  "suggestions": [
    "Start turn-by-turn navigation",
    "Show nearby restaurants"
  ]
}
```

### Example 2: Route Request WITHOUT GPS

**Request:**
```json
POST /api/chat/pure-llm
{
  "message": "Show me route to Blue Mosque",
  "session_id": "user_123"
  // No user_location provided
}
```

**Response:**
```json
{
  "response": "To show you directions, I need your current location. Please enable GPS/location services.",
  "session_id": "user_123",
  "intent": "route_planning",
  "confidence": 1.0,
  "map_data": {
    "request_gps": true,
    "destination": [41.0054, 28.9768]
  },
  "suggestions": [
    "Enable GPS and try again",
    "Specify start location manually"
  ]
}
```

### Example 3: Start Turn-by-Turn Navigation

**Request:**
```json
POST /api/chat/pure-llm
{
  "message": "Navigate to Galata Tower",
  "user_location": {
    "lat": 41.0054,
    "lon": 28.9768
  },
  "session_id": "user_123"
}
```

**Response:**
```json
{
  "response": "🧭 Turn-by-turn navigation started!\n\n➡️ Head north on Alemdar Street\n📍 In 50 meters",
  "session_id": "user_123",
  "intent": "gps_navigation",
  "confidence": 1.0,
  "navigation_active": true,
  "navigation_data": {
    "current_instruction": {
      "text": "Head north on Alemdar Street",
      "distance": 50,
      "type": "continue"
    },
    "progress": {
      "distance_remaining": 2100,
      "time_remaining": 420
    },
    "map_data": {
      "route": [...],
      "current_position": [41.0054, 28.9768]
    }
  }
}
```

## Quick Integration Checklist

### Backend Setup
- [x] Import `get_chat_route_handler()` from `ai_chat_route_integration`
- [x] Pass `user_location` from request to `user_context`
- [x] Call `handle_route_request()` before other processing
- [x] Return `map_data` in response

### Frontend Setup
- [x] Request GPS permission on page load
- [x] Include `user_location` in all chat requests
- [x] Check for `map_data` in responses
- [x] Display route on map when available
- [x] Show navigation button for routes

## Common Patterns

### Pattern 1: Auto-Detect GPS Route
```python
# User says: "How can I go to Taksim?"
# System automatically:
locations = extract_locations("How can I go to Taksim?")  # ["Taksim"]
gps = get_user_gps_location(user_context)                 # (41.0086, 28.9802)
if gps:
    locations.insert(0, gps)  # [(41.0086, 28.9802), "Taksim"]
    route = plan_route(locations[0], locations[1])
```

### Pattern 2: Request GPS Permission
```python
if len(locations) == 1 and not gps_available:
    return {
        'type': 'gps_permission_required',
        'message': 'Please enable GPS',
        'destination': locations[0]
    }
```

### Pattern 3: Display Route on Map
```javascript
function displayRoute(mapData) {
    const route = L.polyline(mapData.route, {color: 'blue'}).addTo(map);
    L.marker(mapData.start).bindPopup('Start').addTo(map);
    L.marker(mapData.end).bindPopup('End').addTo(map);
    map.fitBounds(route.getBounds());
}
```

## Testing

### Quick Test 1: Route with GPS
```bash
cd /Users/omer/Desktop/ai-stanbul
python test_gps_chatbot_integration.py
```

### Quick Test 2: Manual API Test
```bash
curl -X POST http://localhost:8000/api/chat/pure-llm \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How can I go to Taksim?",
    "user_location": {"lat": 41.0086, "lon": 28.9802}
  }'
```

## Supported Queries

### Route Planning
- "How can I go to X?"
- "Show me route to X"
- "Directions to X"
- "How do I get to X?"
- "Take me to X"

### Navigation
- "Navigate to X"
- "Start navigation to X"
- "Stop navigation"
- "What's next?"
- "Where am I?"

## Debug Tips

### Check GPS in Browser Console
```javascript
navigator.geolocation.getCurrentPosition(
    pos => console.log(pos.coords),
    err => console.error(err)
);
```

### Check Backend GPS Extraction
```python
# Add to ai_chat_route_integration.py
logger.info(f"📍 User GPS: {user_context.get('gps')}")
```

### Verify Route Planning
```python
# Test route handler directly
handler = get_chat_route_handler()
result = handler.handle_route_request(
    "How can I go to Taksim?",
    {'gps': {'lat': 41.0086, 'lon': 28.9802}}
)
print(result)
```

## Environment Requirements

- Python 3.8+
- FastAPI
- OSRM routing service
- PostgreSQL
- Modern browser with GPS support
- HTTPS (for GPS in production)

## Quick Links

- 📖 Full Guide: `GPS_TURN_BY_TURN_GUIDE.md`
- ✅ Integration Checklist: `GPS_CHATBOT_INTEGRATION_COMPLETE.md`
- 📋 Final Summary: `GPS_CHATBOT_FINAL_SUMMARY.md`
- 🧪 Test File: `test_gps_chatbot_integration.py`

---

**TL;DR:** User types "How can I go to X?" → System gets GPS → Plans route → Shows on map → Turn-by-turn ready! 🎉
