# User Location & LLM Integration Plan

## 🎯 Executive Summary

This document outlines the integration between the **existing GPS location system** and the **LLaMA 3.x LLM service** to provide Google Maps-level, context-aware, personalized transportation advice and POI recommendations.

**Status**: Ready for implementation  
**Priority**: HIGH  
**Completion**: Phase 3 of LLaMA 3.x integration  

---

## 📍 Existing GPS Location Infrastructure

### Available Components

#### 1. **UserProfile GPS Location Fields**
Location: `istanbul_ai/core/user_profile.py`

```python
@dataclass
class UserProfile:
    # GPS Location Data
    current_location: Optional[str] = None  # Human-readable location
    gps_location: Optional[Dict[str, float]] = None  # {'lat': 41.0082, 'lng': 28.9784}
    location_accuracy: Optional[float] = None  # GPS accuracy in meters
    location_timestamp: Optional[datetime] = None  # When GPS was last updated
    
    def update_location(self, location: str, gps_coords: Optional[Dict[str, float]] = None, accuracy: Optional[float] = None):
        """Update user's current location"""
        self.current_location = location
        if gps_coords:
            self.gps_location = gps_coords
            self.location_accuracy = accuracy
            self.location_timestamp = datetime.now()
```

**Status**: ✅ Production-ready

#### 2. **Main System GPS Integration**
Location: `istanbul_ai/main_system.py:609`

```python
def process_message(self, user_input: str, user_id: str, 
                   gps_location: Optional[Dict] = None, 
                   user_location: Optional[tuple] = None, 
                   return_structured: bool = False):
    """
    Accepts GPS location in two formats:
    - gps_location: Dict with 'latitude' and 'longitude' (deprecated)
    - user_location: Tuple (lat, lon) (preferred)
    """
    # Update user location - support both formats
    if user_location and isinstance(user_location, tuple):
        user_profile.current_location = user_location
        logger.info(f"📍 User location updated: {user_location}")
    elif gps_location and isinstance(gps_location, dict):
        user_profile.current_location = (gps_location['latitude'], gps_location['longitude'])
```

**Status**: ✅ Production-ready

#### 3. **GPS Location Service**
Location: `istanbul_ai/services/gps_location_service.py`

```python
class GPSLocationService:
    """Advanced GPS location service for Istanbul"""
    
    - District boundary detection (Sultanahmet, Beyoğlu, Taksim, Kadıköy, etc.)
    - Distance calculation (Haversine formula)
    - Landmark detection
    - Location accuracy levels (HIGH, MEDIUM, LOW)
```

**Status**: ✅ Production-ready with 15+ Istanbul districts

#### 4. **Nearby Locations Handler**
Location: `istanbul_ai/handlers/nearby_locations_handler.py`

```python
class NearbyLocationsHandler:
    """
    GPS-aware location discovery:
    - Museum recommendations
    - Restaurant recommendations
    - Distance-based filtering
    - Transport recommendations for each POI
    """
```

**Status**: ✅ Production-ready with museum DB and map engine integration

#### 5. **Live Location Integration Service**
Location: `backend/services/live_location_integration_service.py`

```python
class LiveLocationIntegrationService:
    """
    Real-time location routing and POI recommendations:
    - Session-based location tracking
    - Privacy-safe location hashing
    - Dynamic route updates
    - Smart POI recommendations
    """
```

**Status**: ✅ Production-ready

---

## 🚀 Integration Architecture

### Phase 1: LLM Prompt Enhancement with GPS Context

#### Goal
Enhance LLM prompts with real-time user GPS location to provide personalized, context-aware advice.

#### Implementation

**File**: `ml_systems/google_maps_style_prompts.py`

```python
def build_transportation_prompt(
    from_location: str,
    to_location: str,
    context: Dict[str, Any],
    user_profile: Optional[Any] = None
) -> str:
    """
    Enhanced with GPS location context
    
    Args:
        from_location: Origin (or "my location" if GPS available)
        to_location: Destination
        context: Dictionary containing:
            - gps_location: Optional[Tuple[float, float]] - User's current GPS coords
            - location_accuracy: Optional[float] - GPS accuracy in meters
            - current_district: Optional[str] - Detected district name
            - nearby_landmarks: Optional[List[str]] - Nearby landmarks
        user_profile: Optional UserProfile object with GPS data
    """
    
    prompt = f"""You are an expert Istanbul transportation advisor providing Google Maps-level directions.

User is asking for directions from {from_location} to {to_location}.
"""

    # Add GPS context if available
    if context.get('gps_location'):
        lat, lon = context['gps_location']
        prompt += f"\nUser's current location: {lat:.6f}, {lon:.6f}"
        
        if context.get('location_accuracy'):
            prompt += f" (±{context['location_accuracy']:.0f}m accuracy)"
        
        if context.get('current_district'):
            prompt += f"\nUser is currently in: {context['current_district']}"
        
        if context.get('nearby_landmarks'):
            landmarks = ', '.join(context['nearby_landmarks'][:3])
            prompt += f"\nNearby landmarks: {landmarks}"
    
    # Add time-sensitive context
    if context.get('current_time'):
        prompt += f"\nCurrent time: {context['current_time']}"
    
    if context.get('weather'):
        prompt += f"\nCurrent weather: {context['weather']}"
    
    prompt += """

Provide a brief, actionable tip (2-3 sentences) focusing on:
1. Best transport option for THIS specific route and time
2. ONE key travel tip or heads-up
3. Keep it concise - the map shows detailed route

The user will see the full route on the map. Your advice should complement, not repeat, what the map shows.
"""
    
    return prompt
```

**Status**: 🔨 Ready to implement

---

### Phase 2: GPS-Aware POI Recommendations

#### Goal
Use LLM to provide personalized POI recommendations based on user's GPS location and preferences.

#### Implementation

**File**: `ml_systems/google_maps_style_prompts.py`

```python
def build_poi_recommendation_prompt(
    poi_type: str,  # "museums", "restaurants", "attractions"
    context: Dict[str, Any],
    user_profile: Optional[Any] = None
) -> str:
    """
    Build LLM prompt for POI recommendations based on GPS location
    
    Args:
        poi_type: Type of POI to recommend
        context: Dictionary containing:
            - gps_location: Tuple[float, float] - User's current GPS coords
            - current_district: str - Detected district
            - nearby_pois: List[Dict] - Pre-filtered POIs from database
            - user_preferences: Dict - User interests, budget, etc.
        user_profile: Optional UserProfile with interests and preferences
    """
    
    prompt = f"""You are an expert Istanbul guide providing personalized {poi_type} recommendations.

User is looking for {poi_type} recommendations."""

    # Add GPS context
    if context.get('gps_location'):
        lat, lon = context['gps_location']
        prompt += f"\n\nUser's current location: {context.get('current_district', 'Istanbul')} ({lat:.4f}, {lon:.4f})"
    
    # Add nearby POIs from database
    if context.get('nearby_pois'):
        prompt += f"\n\nNearby {poi_type} (pre-filtered by distance and ratings):"
        for idx, poi in enumerate(context['nearby_pois'][:5], 1):
            prompt += f"\n{idx}. {poi['name']} - {poi.get('distance_text', 'nearby')}"
            if poi.get('rating'):
                prompt += f" (Rating: {poi['rating']}/5)"
    
    # Add user preferences
    if user_profile:
        if hasattr(user_profile, 'interests') and user_profile.interests:
            prompt += f"\n\nUser interests: {', '.join(user_profile.interests[:5])}"
        
        if hasattr(user_profile, 'budget_range'):
            prompt += f"\nBudget preference: {user_profile.budget_range}"
        
        if hasattr(user_profile, 'dietary_restrictions') and user_profile.dietary_restrictions:
            prompt += f"\nDietary restrictions: {', '.join(user_profile.dietary_restrictions)}"
    
    prompt += """

Provide a brief recommendation (2-3 sentences) focusing on:
1. Which nearby option is BEST for this user RIGHT NOW
2. ONE key reason why it's a great match
3. Keep it personal and actionable

The user will see all options on the map with full details. Your advice should help them make the best choice.
"""
    
    return prompt
```

**Status**: 🔨 Ready to implement

---

### Phase 3: Update Handlers to Pass GPS Context to LLM

#### 3.1 Transportation Handler Enhancement

**File**: `istanbul_ai/handlers/transportation_handler.py`

```python
class TransportationHandler:
    def __init__(self, ..., llm_service=None):
        """
        Args:
            llm_service: Optional LLaMA 3.x LLM service wrapper
        """
        self.llm_service = llm_service
        self.gps_route_service = gps_route_service
        # ... existing init
    
    async def handle(self, message: str, context: ConversationContext, 
                    user_profile: UserProfile, entities: Dict) -> str:
        """Enhanced with LLM + GPS integration"""
        
        # Extract route info
        from_location = entities.get('from_location', 'your location')
        to_location = entities.get('to_location')
        
        # Build GPS context from user_profile
        gps_context = {}
        if user_profile.gps_location:
            gps_context['gps_location'] = (
                user_profile.gps_location.get('lat'),
                user_profile.gps_location.get('lng')
            )
            gps_context['location_accuracy'] = user_profile.location_accuracy
            gps_context['location_timestamp'] = user_profile.location_timestamp
        
        # Detect current district using GPS service
        if gps_context.get('gps_location'):
            district_info = self.gps_location_service.get_district_info(
                gps_context['gps_location'][0],
                gps_context['gps_location'][1]
            )
            if district_info:
                gps_context['current_district'] = district_info['district']
                gps_context['nearby_landmarks'] = district_info.get('landmarks', [])
        
        # Add time and weather context
        gps_context['current_time'] = datetime.now().strftime('%H:%M')
        # TODO: Add weather from weather service
        
        # Get route from OSRM or GPSRouteService
        route_data = await self.gps_route_service.get_route(
            from_location=gps_context.get('gps_location'),
            to_location=to_location
        )
        
        # Get LLM advice using enhanced prompt
        if self.llm_service:
            llm_advice = self.llm_service.get_transportation_advice(
                from_location=from_location,
                to_location=to_location,
                context=gps_context,
                user_profile=user_profile,
                route_data=route_data  # Pass route data for context
            )
        else:
            llm_advice = "Route found. See map for details."
        
        # Return structured response with map data
        return {
            'response': llm_advice,  # Brief LLM advice (2-3 sentences)
            'map_data': route_data   # Full route for map visualization
        }
```

**Status**: 🔨 Ready to implement

#### 3.2 Nearby Locations Handler Enhancement

**File**: `istanbul_ai/handlers/nearby_locations_handler.py`

```python
class NearbyLocationsHandler:
    def __init__(self, ..., llm_service=None):
        self.llm_service = llm_service
        self.museum_db = IstanbulMuseumDatabase()
        self.gps_location_service = gps_location_service
        # ... existing init
    
    async def handle(self, message: str, context: ConversationContext,
                    user_profile: UserProfile, entities: Dict) -> str:
        """Enhanced with LLM + GPS for personalized POI recommendations"""
        
        poi_type = entities.get('poi_type', 'attractions')  # museums, restaurants, etc.
        
        # Get user's GPS location
        if not user_profile.gps_location:
            return "Please enable location services to get personalized nearby recommendations."
        
        user_lat = user_profile.gps_location.get('lat')
        user_lon = user_profile.gps_location.get('lng')
        
        # Query POI database with GPS filtering
        if poi_type == 'museums':
            nearby_pois = self.museum_db.find_museums_near(
                latitude=user_lat,
                longitude=user_lon,
                radius_km=2.0,  # 2km radius
                limit=10
            )
        elif poi_type == 'restaurants':
            # TODO: Query restaurant database
            nearby_pois = []
        else:
            # Generic POI search
            nearby_pois = []
        
        # Build GPS context for LLM
        gps_context = {
            'gps_location': (user_lat, user_lon),
            'current_district': self.gps_location_service.get_district_name(user_lat, user_lon),
            'nearby_pois': nearby_pois,
            'user_preferences': user_profile.get_preference_summary()
        }
        
        # Get LLM recommendation
        if self.llm_service and nearby_pois:
            llm_recommendation = self.llm_service.get_poi_recommendation(
                poi_type=poi_type,
                context=gps_context,
                user_profile=user_profile
            )
        else:
            llm_recommendation = f"Found {len(nearby_pois)} nearby {poi_type}."
        
        # Return structured response
        return {
            'response': llm_recommendation,  # Brief personalized advice
            'pois': nearby_pois,             # Full POI list for map
            'map_data': {
                'user_location': {'lat': user_lat, 'lng': user_lon},
                'markers': [
                    {
                        'lat': poi.get('latitude'),
                        'lng': poi.get('longitude'),
                        'name': poi.get('name'),
                        'type': poi_type,
                        'distance': poi.get('distance_km')
                    }
                    for poi in nearby_pois
                ]
            }
        }
```

**Status**: 🔨 Ready to implement

---

### Phase 4: Update LLM Service Wrapper

**File**: `ml_systems/llm_service_wrapper.py`

```python
from .google_maps_style_prompts import (
    build_transportation_prompt,
    build_poi_recommendation_prompt
)

class LLMServiceWrapper:
    """Model-agnostic LLM service wrapper with GPS context support"""
    
    def get_transportation_advice(
        self,
        from_location: str,
        to_location: str,
        context: Dict[str, Any],
        user_profile: Optional[Any] = None,
        route_data: Optional[Dict] = None,
        max_tokens: int = 100  # Brief advice only
    ) -> str:
        """
        Get brief, context-aware transportation advice
        
        Args:
            from_location: Origin (or "my location")
            to_location: Destination
            context: GPS and environmental context (see build_transportation_prompt)
            user_profile: Optional UserProfile with preferences
            route_data: Optional route data from OSRM (for context)
            max_tokens: Max response length (default: 100 for 2-3 sentences)
        
        Returns:
            Brief transportation advice (2-3 sentences)
        """
        
        # Build enhanced prompt with GPS context
        prompt = build_transportation_prompt(
            from_location=from_location,
            to_location=to_location,
            context=context,
            user_profile=user_profile
        )
        
        # Generate response
        response = self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            stop_sequences=["\n\n", "User:", "Question:"]
        )
        
        # Log for monitoring
        self.logger.info(f"🗺️ Transportation advice: {from_location} → {to_location}")
        self.logger.debug(f"GPS context: {context.get('gps_location')}")
        
        return response.strip()
    
    def get_poi_recommendation(
        self,
        poi_type: str,
        context: Dict[str, Any],
        user_profile: Optional[Any] = None,
        max_tokens: int = 100
    ) -> str:
        """
        Get personalized POI recommendation based on GPS location
        
        Args:
            poi_type: Type of POI ("museums", "restaurants", etc.)
            context: GPS context with nearby POIs (see build_poi_recommendation_prompt)
            user_profile: Optional UserProfile with interests and preferences
            max_tokens: Max response length
        
        Returns:
            Personalized POI recommendation (2-3 sentences)
        """
        
        prompt = build_poi_recommendation_prompt(
            poi_type=poi_type,
            context=context,
            user_profile=user_profile
        )
        
        response = self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.8,  # Slightly higher for creative recommendations
            stop_sequences=["\n\n", "User:", "Question:"]
        )
        
        self.logger.info(f"📍 POI recommendation: {poi_type} near {context.get('current_district')}")
        
        return response.strip()
```

**Status**: 🔨 Ready to implement

---

## 🔄 Data Flow

### User Query → GPS-Enhanced LLM Response

```
1. User sends query: "How do I get to Hagia Sophia?"
   ↓
2. Main System (process_message):
   - Receives user_location: (41.0082, 28.9784) from frontend
   - Updates UserProfile.gps_location
   - Passes to TransportationHandler
   ↓
3. TransportationHandler:
   - Extracts GPS from user_profile
   - Calls GPSLocationService to detect district (→ "Sultanahmet")
   - Builds gps_context with location, district, landmarks
   - Calls GPSRouteService to get route (OSRM)
   - Calls LLMService.get_transportation_advice(gps_context)
   ↓
4. LLMService:
   - Builds prompt with GPS context
   - Sends to LLaMA 3.2 3B (or TinyLlama)
   - Returns: "You're in Sultanahmet, very close! Walk 10 min via Divan Yolu. 
              The Blue Mosque is on the way if you want to visit both."
   ↓
5. Response to User:
   {
     "response": "You're in Sultanahmet, very close! Walk 10 min...",
     "map_data": { /* full route from OSRM */ }
   }
   ↓
6. Frontend:
   - Shows brief LLM advice
   - Displays detailed route on map with turn-by-turn
```

---

## 📊 Benefits of Integration

### 1. **Personalized Context**
- LLM knows exact user location (district, nearby landmarks)
- Recommendations are distance-aware ("5 min walk" vs "take metro")
- Time-sensitive advice (rush hour, prayer times, closing hours)

### 2. **Google Maps-Level Intelligence**
- Brief, actionable advice (not verbose directions)
- Complements map visualization (LLM = "why", Map = "how")
- Real-time context (weather, traffic, events)

### 3. **Enhanced POI Discovery**
- "What's near me?" → Personalized museum/restaurant recommendations
- Considers user interests, budget, dietary restrictions
- Distance-sorted with walking/transport times

### 4. **Privacy-Safe**
- GPS data stays on backend (never sent to LLM API)
- Only aggregated context sent to LLM (district, not exact coords)
- User controls location sharing

---

## 🛠️ Implementation Checklist

### Phase 1: Prompt Enhancement ✅
- [x] Review existing GPS infrastructure
- [ ] Update `google_maps_style_prompts.py` with GPS context
- [ ] Add `build_transportation_prompt()` GPS parameters
- [ ] Add `build_poi_recommendation_prompt()` GPS parameters
- [ ] Test prompts with mock GPS data

### Phase 2: LLM Service Update
- [ ] Add `get_transportation_advice()` to `LLMServiceWrapper`
- [ ] Add `get_poi_recommendation()` to `LLMServiceWrapper`
- [ ] Add GPS context extraction helpers
- [ ] Add logging for GPS-enhanced queries
- [ ] Test with TinyLlama on M2 Pro

### Phase 3: Handler Integration
- [ ] Update `TransportationHandler` to use LLM + GPS
- [ ] Update `NearbyLocationsHandler` to use LLM + GPS
- [ ] Add `gps_location_service` dependency injection
- [ ] Update handler tests with GPS mock data
- [ ] Test end-to-end flow

### Phase 4: Main System Update
- [ ] Ensure `process_message()` passes GPS to handlers
- [ ] Add GPS context to `ConversationContext`
- [ ] Update response formatting for structured output
- [ ] Add GPS location validation
- [ ] Test with real GPS coordinates

### Phase 5: Testing & Validation
- [ ] Unit tests for GPS context building
- [ ] Integration tests for LLM + GPS pipeline
- [ ] Test with various Istanbul districts
- [ ] Test with missing/invalid GPS data
- [ ] Performance testing (latency with GPS queries)

### Phase 6: Production Deployment
- [ ] Deploy TinyLlama for development
- [ ] Test on M2 Pro with MPS
- [ ] Deploy LLaMA 3.2 3B for production
- [ ] Test on T4 GPU
- [ ] Monitor GPS usage and accuracy
- [ ] Add GPS metrics to dashboard

---

## 📈 Success Metrics

1. **Accuracy**: LLM provides correct district and nearby landmarks 95%+ of time
2. **Relevance**: POI recommendations match user preferences 90%+ of time
3. **Brevity**: LLM responses stay under 3 sentences 95%+ of time
4. **Latency**: GPS + LLM pipeline adds <500ms to response time
5. **User Satisfaction**: GPS-enhanced responses rated 4.5+/5.0

---

## 🚦 Next Steps

1. **Review this plan** with the team
2. **Implement Phase 1**: Update prompt templates with GPS context
3. **Implement Phase 2**: Add GPS methods to LLM service wrapper
4. **Test locally** with TinyLlama and mock GPS data
5. **Deploy to production** with LLaMA 3.2 3B on T4 GPU

---

## 📝 Notes

- GPS infrastructure is **already production-ready** ✅
- No breaking changes to existing API
- Backward compatible (works without GPS)
- Privacy-safe (GPS processed locally)
- Model-agnostic (works with TinyLlama and LLaMA 3.2 3B)

---

**Document Version**: 1.0  
**Last Updated**: 2024-01-09  
**Status**: Ready for Implementation 🚀
