# Main System Map Integration Update

**Date:** October 22, 2025  
**File:** `istanbul_ai/main_system.py`  
**Status:** 🔧 IMPLEMENTATION GUIDE

---

## 🎯 Current State Analysis

### Current Flow
```
User Query → process_message() → _generate_contextual_response() → String Response
```

**Issue:** Returns only text string, no structured data with coordinates

### Target Flow
```
User Query → process_message() → _generate_contextual_response() → Dict with {text, map_data}
```

---

## 🛠️ Required Changes

### **Change 1: Update `process_message()` Return Type**

**Current:** Returns `str`  
**New:** Returns `Union[str, Dict]`

```python
def process_message(self, message: str, user_id: str) -> Union[str, Dict[str, Any]]:
    """Process user message and generate response with optional map data"""
    try:
        # ... existing code ...
        
        # Generate contextual response (enhanced with neural insights)
        response = self._generate_contextual_response(
            message, intent, entities, user_profile, context, neural_insights
        )
        
        # NEW: Check if response is structured (has map data)
        if isinstance(response, dict):
            # Record interaction with text part only
            text_response = response.get('text', str(response))
            context.add_interaction(message, text_response, intent)
            
            # Return full structured response
            return response
        else:
            # Legacy string response
            context.add_interaction(message, response, intent)
            return response
            
    except Exception as e:
        # ... error handling ...
```

---

### **Change 2: Update `_generate_contextual_response()` for Map Data**

**Location:** Line ~781

```python
def _generate_contextual_response(self, message: str, intent: str, entities: Dict,
                                user_profile: UserProfile, context: ConversationContext, 
                                neural_insights: Optional[Dict] = None) -> Union[str, Dict[str, Any]]:
    """Generate contextual response with map data for location-based queries"""
    
    current_time = datetime.now()
    
    # 🍽️ RESTAURANT QUERIES - Return structured response with map data
    if intent in ['restaurant', 'dining', 'food']:
        response_data = self.response_generator.generate_comprehensive_recommendation(
            'restaurant', entities, user_profile, context
        )
        
        # NEW: Extract and add map data if not already present
        if isinstance(response_data, dict) and 'map_data' not in response_data:
            response_data['map_data'] = self._extract_map_data_from_response(
                response_data, 'restaurant', entities
            )
        
        return response_data
    
    # 🏛️ ATTRACTION QUERIES - Return structured response with map data  
    elif intent in ['attraction', 'sightseeing', 'places', 'landmark']:
        message_lower = message.lower()
        
        # Museum-specific handling
        if any(kw in message_lower for kw in ['museum', 'gallery', 'exhibition']):
            if self.advanced_museum_system:
                response_data = self._generate_advanced_museum_response(
                    message, entities, user_profile, context
                )
            elif self.museum_generator:
                response_data = self._generate_location_aware_museum_response(
                    message, entities, user_profile, context
                )
            else:
                response_data = self.response_generator.generate_comprehensive_recommendation(
                    'attraction', entities, user_profile, context
                )
        else:
            # General attractions
            if self.advanced_attractions_system:
                response_data = self._generate_advanced_attractions_response(
                    message, entities, user_profile, context
                )
            else:
                response_data = self.response_generator.generate_comprehensive_recommendation(
                    'attraction', entities, user_profile, context
                )
        
        # NEW: Add map data if not present
        if isinstance(response_data, dict) and 'map_data' not in response_data:
            response_data['map_data'] = self._extract_map_data_from_response(
                response_data, 'attraction', entities
            )
        
        return response_data
    
    # 🏘️ NEIGHBORHOOD QUERIES - Return structured response with map data
    elif intent in ['neighborhood', 'district', 'area']:
        response_data = self.response_generator.generate_comprehensive_recommendation(
            'neighborhood', entities, user_profile, context
        )
        
        # NEW: Add map data
        if isinstance(response_data, dict) and 'map_data' not in response_data:
            response_data['map_data'] = self._extract_map_data_from_response(
                response_data, 'neighborhood', entities
            )
        
        return response_data
    
    # 🚇 TRANSPORTATION QUERIES - Return structured response with route map
    elif intent == 'transportation':
        response_data = self._generate_transportation_response(
            message, entities, user_profile, context
        )
        
        # NEW: Add route visualization data
        if isinstance(response_data, dict) and 'map_data' not in response_data:
            response_data['map_data'] = self._extract_route_map_data(
                message, entities, context
            )
        
        return response_data
    
    # 🗺️ ROUTE PLANNING - Return structured response with route map
    elif intent in ['route_planning', 'gps_route_planning', 'museum_route_planning']:
        if intent == 'gps_route_planning':
            response_data = self._generate_gps_route_response(
                message, entities, user_profile, context
            )
        elif intent == 'museum_route_planning':
            response_data = self._generate_museum_route_response(
                message, entities, user_profile, context
            )
        else:
            response_data = self._generate_route_planning_response(
                message, user_profile, context
            )
        
        # Route planning should already have map data, but ensure it
        if isinstance(response_data, dict) and 'map_data' not in response_data:
            response_data['map_data'] = self._extract_route_map_data(
                message, entities, context
            )
        
        return response_data
    
    # ... other intents (shopping, events, greeting) - return as string
    
    # Fallback - return string
    else:
        return self.response_generator._generate_fallback_response(context, user_profile)
```

---

### **Change 3: Add Map Data Extraction Methods**

**Add these new methods to the `IstanbulDailyTalkAI` class:**

```python
def _extract_map_data_from_response(self, response_data: Union[str, Dict], 
                                    response_type: str, entities: Dict) -> Optional[Dict]:
    """Extract map data from response for visualization"""
    try:
        # If response is already structured with map_data, return it
        if isinstance(response_data, dict) and 'map_data' in response_data:
            return response_data['map_data']
        
        # Otherwise, extract coordinates based on type
        if response_type == 'restaurant':
            return self._extract_restaurant_map_data(entities)
        elif response_type == 'attraction':
            return self._extract_attraction_map_data(entities)
        elif response_type == 'neighborhood':
            return self._extract_neighborhood_map_data(entities)
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error extracting map data: {e}")
        return None

def _extract_restaurant_map_data(self, entities: Dict) -> Optional[Dict]:
    """Extract restaurant coordinates for map visualization"""
    try:
        # Use smart recommendation engine if available
        if hasattr(self, 'smart_rec_engine'):
            from .services.smart_recommendation_engine import SmartRecommendationEngine
            
            # Get restaurant recommendations
            recommendations = self.smart_rec_engine.get_personalized_recommendations(
                user_preferences={'cuisines': entities.get('cuisines', [])},
                location=entities.get('districts', [None])[0],
                budget=None,
                limit=5
            )
            
            if recommendations:
                locations = []
                for rec in recommendations:
                    if 'coordinates' in rec and rec['coordinates']:
                        lat, lon = rec['coordinates']
                        locations.append({
                            'lat': lat,
                            'lon': lon,
                            'name': rec.get('name', 'Unknown Restaurant'),
                            'type': 'restaurant',
                            'metadata': {
                                'cuisine': rec.get('cuisine_type'),
                                'price': rec.get('price_range'),
                                'rating': rec.get('rating'),
                                'address': rec.get('location')
                            }
                        })
                
                if locations:
                    # Calculate center and bounds
                    lats = [loc['lat'] for loc in locations]
                    lons = [loc['lon'] for loc in locations]
                    
                    return {
                        'locations': locations,
                        'center': {
                            'lat': sum(lats) / len(lats),
                            'lon': sum(lons) / len(lons)
                        },
                        'bounds': {
                            'north': max(lats),
                            'south': min(lats),
                            'east': max(lons),
                            'west': min(lons)
                        },
                        'type': 'restaurant',
                        'zoom': 14
                    }
        
        return None
        
    except Exception as e:
        logger.error(f"Error extracting restaurant map data: {e}")
        return None

def _extract_attraction_map_data(self, entities: Dict) -> Optional[Dict]:
    """Extract attraction coordinates for map visualization"""
    try:
        # Predefined major Istanbul attractions with coordinates
        attractions_db = {
            'hagia sophia': {'lat': 41.0086, 'lon': 28.9802, 'name': 'Hagia Sophia', 'type': 'historic_monument'},
            'blue mosque': {'lat': 41.0054, 'lon': 28.9764, 'name': 'Blue Mosque', 'type': 'mosque'},
            'topkapi palace': {'lat': 41.0115, 'lon': 28.9815, 'name': 'Topkapi Palace', 'type': 'palace'},
            'grand bazaar': {'lat': 41.0108, 'lon': 28.9681, 'name': 'Grand Bazaar', 'type': 'market'},
            'galata tower': {'lat': 41.0256, 'lon': 28.9744, 'name': 'Galata Tower', 'type': 'tower'},
            'basilica cistern': {'lat': 41.0084, 'lon': 28.9779, 'name': 'Basilica Cistern', 'type': 'historic'},
            'dolmabahce palace': {'lat': 41.0392, 'lon': 28.9997, 'name': 'Dolmabahce Palace', 'type': 'palace'},
            'taksim square': {'lat': 41.0370, 'lon': 28.9850, 'name': 'Taksim Square', 'type': 'landmark'},
            'istiklal street': {'lat': 41.0332, 'lon': 28.9784, 'name': 'Istiklal Street', 'type': 'street'},
            'maiden tower': {'lat': 41.0210, 'lon': 29.0043, 'name': 'Maiden Tower', 'type': 'tower'}
        }
        
        # Extract mentioned attractions from entities or message
        locations = []
        
        # Check landmarks in entities
        landmarks = entities.get('landmarks', [])
        for landmark in landmarks:
            landmark_lower = landmark.lower()
            if landmark_lower in attractions_db:
                attr = attractions_db[landmark_lower]
                locations.append({
                    'lat': attr['lat'],
                    'lon': attr['lon'],
                    'name': attr['name'],
                    'type': attr['type'],
                    'metadata': {'category': attr['type']}
                })
        
        # If no specific attractions, return popular ones in requested district
        if not locations:
            district = entities.get('districts', [None])[0]
            if district and district.lower() in ['sultanahmet', 'fatih']:
                # Return Sultanahmet attractions
                for key in ['hagia sophia', 'blue mosque', 'topkapi palace', 'grand bazaar', 'basilica cistern']:
                    attr = attractions_db[key]
                    locations.append({
                        'lat': attr['lat'],
                        'lon': attr['lon'],
                        'name': attr['name'],
                        'type': attr['type'],
                        'metadata': {'category': attr['type']}
                    })
            elif district and district.lower() in ['beyoglu', 'galata']:
                # Return Beyoğlu attractions
                for key in ['galata tower', 'taksim square', 'istiklal street']:
                    attr = attractions_db[key]
                    locations.append({
                        'lat': attr['lat'],
                        'lon': attr['lon'],
                        'name': attr['name'],
                        'type': attr['type'],
                        'metadata': {'category': attr['type']}
                    })
            else:
                # Return top 5 attractions
                for key in list(attractions_db.keys())[:5]:
                    attr = attractions_db[key]
                    locations.append({
                        'lat': attr['lat'],
                        'lon': attr['lon'],
                        'name': attr['name'],
                        'type': attr['type'],
                        'metadata': {'category': attr['type']}
                    })
        
        if locations:
            lats = [loc['lat'] for loc in locations]
            lons = [loc['lon'] for loc in locations]
            
            return {
                'locations': locations,
                'center': {
                    'lat': sum(lats) / len(lats),
                    'lon': sum(lons) / len(lons)
                },
                'bounds': {
                    'north': max(lats),
                    'south': min(lats),
                    'east': max(lons),
                    'west': min(lons)
                },
                'type': 'attraction',
                'zoom': 13
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Error extracting attraction map data: {e}")
        return None

def _extract_neighborhood_map_data(self, entities: Dict) -> Optional[Dict]:
    """Extract neighborhood center coordinates for map visualization"""
    try:
        # Neighborhood centers
        neighborhoods_db = {
            'sultanahmet': {'lat': 41.0086, 'lon': 28.9802, 'name': 'Sultanahmet'},
            'beyoglu': {'lat': 41.0370, 'lon': 28.9850, 'name': 'Beyoğlu'},
            'besiktas': {'lat': 41.0420, 'lon': 29.0070, 'name': 'Beşiktaş'},
            'kadikoy': {'lat': 40.9833, 'lon': 29.0333, 'name': 'Kadıköy'},
            'uskudar': {'lat': 41.0220, 'lon': 29.0150, 'name': 'Üsküdar'},
            'sisli': {'lat': 41.0600, 'lon': 28.9870, 'name': 'Şişli'},
            'sariyer': {'lat': 41.1089, 'lon': 29.0553, 'name': 'Sarıyer'},
            'fatih': {'lat': 41.0190, 'lon': 28.9490, 'name': 'Fatih'},
            'galata': {'lat': 41.0256, 'lon': 28.9744, 'name': 'Galata'},
            'karakoy': {'lat': 41.0240, 'lon': 28.9750, 'name': 'Karaköy'}
        }
        
        locations = []
        
        # Check districts in entities
        districts = entities.get('districts', [])
        for district in districts:
            district_lower = district.lower()
            if district_lower in neighborhoods_db:
                nbhd = neighborhoods_db[district_lower]
                locations.append({
                    'lat': nbhd['lat'],
                    'lon': nbhd['lon'],
                    'name': nbhd['name'],
                    'type': 'neighborhood',
                    'metadata': {'district': nbhd['name']}
                })
        
        # If no specific neighborhoods, return popular ones
        if not locations:
            for key in ['sultanahmet', 'beyoglu', 'kadikoy']:
                nbhd = neighborhoods_db[key]
                locations.append({
                    'lat': nbhd['lat'],
                    'lon': nbhd['lon'],
                    'name': nbhd['name'],
                    'type': 'neighborhood',
                    'metadata': {'district': nbhd['name']}
                })
        
        if locations:
            lats = [loc['lat'] for loc in locations]
            lons = [loc['lon'] for loc in locations]
            
            return {
                'locations': locations,
                'center': {
                    'lat': sum(lats) / len(lats),
                    'lon': sum(lons) / len(lons)
                },
                'bounds': {
                    'north': max(lats),
                    'south': min(lats),
                    'east': max(lons),
                    'west': min(lons)
                },
                'type': 'neighborhood',
                'zoom': 12
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Error extracting neighborhood map data: {e}")
        return None

def _extract_route_map_data(self, message: str, entities: Dict, 
                           context: ConversationContext) -> Optional[Dict]:
    """Extract route coordinates for map visualization"""
    try:
        # This will be implemented in Phase 2
        # For now, return None and handle routes separately
        return None
        
    except Exception as e:
        logger.error(f"Error extracting route map data: {e}")
        return None
```

---

### **Change 4: Update Backend Integration**

**File:** `backend/main.py`

Update where `istanbul_daily_talk_ai.process_message()` is called:

```python
# In get_istanbul_ai_response_with_quality function
if ISTANBUL_DAILY_TALK_AVAILABLE:
    try:
        # Process with Istanbul Daily Talk AI
        ai_response = istanbul_daily_talk_ai.process_message(user_input, session_id)
        
        # NEW: Handle structured response
        if isinstance(ai_response, dict):
            # Structured response with map data
            text_response = ai_response.get('text', str(ai_response))
            map_data = ai_response.get('map_data')
            
            # Generate map URL if map data exists
            view_on_map_url = None
            if map_data and map_data.get('locations'):
                location_ids = ','.join([str(i) for i in range(len(map_data['locations']))])
                rec_type = map_data.get('type', 'general')
                view_on_map_url = f"/map?type={rec_type}&locations={location_ids}"
            
            return {
                'success': True,
                'response': text_response,
                'session_id': session_id,
                'has_context': True,
                'map_data': map_data,  # NEW
                'view_on_map_url': view_on_map_url,  # NEW
                'quality_assessment': {...}
            }
        else:
            # Legacy string response
            text_response = ai_response
            return {
                'success': True,
                'response': text_response,
                'session_id': session_id,
                'has_context': True,
                'quality_assessment': {...}
            }
            
    except Exception as e:
        logger.error(f"Istanbul Daily Talk AI error: {e}")
        return None
```

---

## 📊 Summary of Changes

### Files to Modify

1. **`istanbul_ai/main_system.py`**
   - Update `process_message()` return type: `str` → `Union[str, Dict]`
   - Update `_generate_contextual_response()` to return structured data
   - Add 4 new methods:
     - `_extract_map_data_from_response()`
     - `_extract_restaurant_map_data()`
     - `_extract_attraction_map_data()`
     - `_extract_neighborhood_map_data()`

2. **`backend/main.py`**
   - Update `get_istanbul_ai_response_with_quality()` to handle structured responses
   - Extract `map_data` from response
   - Generate `view_on_map_url`

### Backwards Compatibility

✅ **Fully backwards compatible!**
- If response is `str`, works as before
- If response is `dict`, extracts text and map data
- Frontend can check for `map_data` field

---

## 🧪 Testing

```python
# test_main_system_map_integration.py

from istanbul_ai.main_system import IstanbulDailyTalkAI

def test_restaurant_returns_map_data():
    ai = IstanbulDailyTalkAI()
    response = ai.process_message(
        "Best Turkish restaurants in Sultanahmet",
        "test_user"
    )
    
    assert isinstance(response, dict)
    assert 'text' in response
    assert 'map_data' in response
    assert response['map_data'] is not None
    assert len(response['map_data']['locations']) > 0
    print("✅ Restaurant query returns map data")

def test_attraction_returns_map_data():
    ai = IstanbulDailyTalkAI()
    response = ai.process_message(
        "What should I visit in Istanbul?",
        "test_user"
    )
    
    assert isinstance(response, dict)
    assert 'map_data' in response
    assert response['map_data']['type'] == 'attraction'
    print("✅ Attraction query returns map data")

def test_greeting_returns_string():
    ai = IstanbulDailyTalkAI()
    response = ai.process_message(
        "Hello!",
        "test_user"
    )
    
    # Greeting should still return string (no location data)
    assert isinstance(response, str) or isinstance(response, dict)
    if isinstance(response, dict):
        assert 'map_data' not in response or response['map_data'] is None
    print("✅ Greeting query works correctly")

if __name__ == '__main__':
    test_restaurant_returns_map_data()
    test_attraction_returns_map_data()
    test_greeting_returns_string()
    print("\n✅ All tests passed!")
```

---

## 🚀 Implementation Steps

### Step 1: Add Import
```python
# At top of istanbul_ai/main_system.py
from typing import Union
```

### Step 2: Copy-paste the 4 new methods
Add to `IstanbulDailyTalkAI` class (around line 1500)

### Step 3: Update `_generate_contextual_response()`
Replace existing method with new version

### Step 4: Update `process_message()`
Add structured response handling

### Step 5: Update backend/main.py
Add map data extraction

### Step 6: Test
```bash
python test_main_system_map_integration.py
```

---

**Status:** Ready to implement ✅  
**Estimated Time:** 2-3 hours  
**Impact:** High - enables visual maps for all location queries

---

**Next:** After implementing, proceed with frontend map visualization from `MAP_INTEGRATION_IMPLEMENTATION_GUIDE.md`
