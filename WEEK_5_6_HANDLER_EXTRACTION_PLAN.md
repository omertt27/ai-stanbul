# Week 5-6: Handler Layer Extraction Plan

**Date Started:** October 29, 2025  
**Estimated Time:** 12 hours  
**Status:** 🚀 STARTING

---

## 🎯 Goals

Extract handler logic from `main_system.py` into dedicated handler modules:

1. **DailyTalkHandler** - Handle casual conversation and greetings
2. **AttractionHandler** - Handle attraction/museum queries
3. **RestaurantHandler** - Handle restaurant queries
4. **TransportationHandler** - Handle transportation queries
5. **EventHandler** - Handle event queries
6. **WeatherHandler** - Handle weather queries
7. **NeighborhoodHandler** - Handle neighborhood queries

---

## 📋 Current State Analysis

### Main System Stats
- **Current Lines:** ~2,477 lines
- **Response Generation Methods:** ~800 lines
- **Handler Logic:** ~600 lines
- **Target Reduction:** -1,400 lines (down to ~1,077 lines)

### Methods to Extract

From `main_system.py`:
- `_generate_restaurant_response()` - ~150 lines → RestaurantHandler
- `_generate_attraction_response()` - ~200 lines → AttractionHandler
- `_generate_transportation_response()` - ~180 lines → TransportationHandler
- `_generate_event_response()` - ~120 lines → EventHandler
- `_generate_weather_response()` - ~100 lines → WeatherHandler
- `_generate_neighborhood_response()` - ~90 lines → NeighborhoodHandler
- Daily talk methods - ~160 lines → DailyTalkHandler

---

## 🏗️ Implementation Plan

### Phase 1: Create Handler Base Class (1 hour)

**File:** `istanbul_ai/handlers/base_handler.py`

```python
"""
Base Handler - Abstract base class for all handlers
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from ..core.models import UserProfile, ConversationContext


class BaseHandler(ABC):
    """Abstract base class for all response handlers"""
    
    def __init__(self):
        """Initialize the handler"""
        self.handler_name = self.__class__.__name__
    
    @abstractmethod
    def can_handle(self, intent: str, entities: Dict, context: ConversationContext) -> bool:
        """
        Determine if this handler can handle the query
        
        Args:
            intent: Classified intent
            entities: Extracted entities
            context: Conversation context
        
        Returns:
            True if this handler can handle the query
        """
        pass
    
    @abstractmethod
    def handle(
        self, 
        message: str, 
        intent: str, 
        entities: Dict,
        user_profile: UserProfile,
        context: ConversationContext
    ) -> str:
        """
        Handle the query and generate response
        
        Args:
            message: User message
            intent: Classified intent
            entities: Extracted entities
            user_profile: User profile
            context: Conversation context
        
        Returns:
            Response string
        """
        pass
    
    def _ensure_language(self, response: str, user_profile: UserProfile) -> str:
        """
        Ensure response is in user's preferred language
        
        Args:
            response: Generated response
            user_profile: User profile
        
        Returns:
            Response in correct language
        """
        # Language switching logic
        preferred_lang = user_profile.preferred_language or 'en'
        
        # If response doesn't match preferred language, translate
        # This will be implemented by each handler
        
        return response
```

### Phase 2: Extract DailyTalkHandler (2 hours)

**File:** `istanbul_ai/handlers/daily_talk_handler.py`

**What to Extract:**
- Daily talk detection logic
- Greeting responses
- Thanks responses
- Goodbye responses
- Weather small talk
- Basic conversation

**Dependencies:**
- Enhanced Bilingual Daily Talks System
- ML-Enhanced Daily Talks Bridge
- Basic bilingual responder (fallback)

**Lines to Move:** ~160 lines from main_system.py

### Phase 3: Extract RestaurantHandler (2 hours)

**File:** `istanbul_ai/handlers/restaurant_handler.py`

**What to Extract:**
- Restaurant query processing
- Cuisine filtering
- District filtering
- Price range filtering
- Restaurant recommendations
- Response formatting

**Dependencies:**
- Restaurant database service
- Price filter service
- Hidden gems handler (for local restaurants)

**Lines to Move:** ~150 lines from main_system.py

### Phase 4: Extract AttractionHandler (2 hours)

**File:** `istanbul_ai/handlers/attraction_handler.py`

**What to Extract:**
- Attraction/museum query processing
- Landmark recommendations
- Historical site information
- Museum information
- District-based attractions
- Response formatting

**Dependencies:**
- Attractions system
- Museum system
- Hidden gems handler

**Lines to Move:** ~200 lines from main_system.py

### Phase 5: Extract TransportationHandler (2 hours)

**File:** `istanbul_ai/handlers/transportation_handler.py`

**What to Extract:**
- Transportation query processing
- Metro/bus/ferry information
- Route planning
- Transfer instructions
- Airport transportation
- Response formatting

**Dependencies:**
- Transportation system
- Route planner
- Transfer instructions generator

**Lines to Move:** ~180 lines from main_system.py

### Phase 6: Extract EventHandler (1 hour)

**File:** `istanbul_ai/handlers/event_handler.py`

**What to Extract:**
- Event query processing
- Concert/show recommendations
- Cultural event information
- Festival information
- Response formatting

**Dependencies:**
- Events service
- İKSV events system

**Lines to Move:** ~120 lines from main_system.py

### Phase 7: Extract WeatherHandler (1 hour)

**File:** `istanbul_ai/handlers/weather_handler.py`

**What to Extract:**
- Weather query processing
- Weather recommendations
- Seasonal guidance
- Activity suggestions based on weather
- Response formatting

**Dependencies:**
- Weather service

**Lines to Move:** ~100 lines from main_system.py

### Phase 8: Extract NeighborhoodHandler (1 hour)

**File:** `istanbul_ai/handlers/neighborhood_handler.py`

**What to Extract:**
- Neighborhood query processing
- Area recommendations
- District characteristics
- Response formatting

**Dependencies:**
- Neighborhood data
- Hidden gems handler

**Lines to Move:** ~90 lines from main_system.py

---

## 🧪 Testing Strategy

### For Each Handler

1. **Unit Tests** (`tests/handlers/test_<handler_name>.py`)
   - Test can_handle() logic
   - Test handle() with various inputs
   - Test edge cases
   - Test error handling

2. **Integration Tests**
   - Test handler with real services
   - Test with main system
   - Test response quality

3. **Performance Tests**
   - Response time < 200ms
   - Memory usage stable

### Test Template

```python
import unittest
from istanbul_ai.handlers.<handler_name> import <HandlerClass>
from istanbul_ai.core.models import UserProfile, ConversationContext


class Test<HandlerClass>(unittest.TestCase):
    
    def setUp(self):
        self.handler = <HandlerClass>()
        self.user_profile = UserProfile(user_id="test_user")
        self.context = ConversationContext(session_id="test", user_profile=self.user_profile)
    
    def test_can_handle_positive(self):
        """Test handler correctly identifies queries it can handle"""
        intent = '<intent_type>'
        entities = {}
        
        result = self.handler.can_handle(intent, entities, self.context)
        
        self.assertTrue(result)
    
    def test_can_handle_negative(self):
        """Test handler correctly rejects queries it cannot handle"""
        intent = 'other_intent'
        entities = {}
        
        result = self.handler.can_handle(intent, entities, self.context)
        
        self.assertFalse(result)
    
    def test_handle_basic_query(self):
        """Test handler generates appropriate response"""
        message = "test query"
        intent = '<intent_type>'
        entities = {}
        
        response = self.handler.handle(message, intent, entities, self.user_profile, self.context)
        
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
```

---

## 📊 Success Metrics

### Code Quality
- ✅ Each handler < 300 lines
- ✅ Test coverage > 80% per handler
- ✅ All handlers inherit from BaseHandler
- ✅ Clear separation of concerns

### Performance
- ✅ No performance regression
- ✅ Response time < 200ms per handler
- ✅ Memory usage stable

### Functionality
- ✅ All existing tests pass
- ✅ All handlers properly integrated
- ✅ Response quality maintained

---

## 🔄 Integration with Main System

After extraction, `main_system.py` will delegate to handlers:

```python
from .handlers import (
    DailyTalkHandler,
    RestaurantHandler,
    AttractionHandler,
    TransportationHandler,
    EventHandler,
    WeatherHandler,
    NeighborhoodHandler
)


class IstanbulDailyTalkAI:
    
    def __init__(self):
        # ...existing initialization...
        
        # Initialize handlers
        self.handlers = {
            'daily_talk': DailyTalkHandler(),
            'restaurant': RestaurantHandler(),
            'attraction': AttractionHandler(),
            'transportation': TransportationHandler(),
            'event': EventHandler(),
            'weather': WeatherHandler(),
            'neighborhood': NeighborhoodHandler()
        }
    
    def process_message(self, message: str, user_id: str) -> str:
        # ...existing preprocessing...
        
        # Find appropriate handler
        for handler_name, handler in self.handlers.items():
            if handler.can_handle(intent, entities, context):
                response = handler.handle(
                    message, intent, entities, user_profile, context
                )
                return response
        
        # Default response if no handler matches
        return self._generate_default_response(message)
```

---

## 📝 Implementation Checklist

### Phase 1: Base Handler
- [ ] Create `handlers/__init__.py`
- [ ] Create `base_handler.py`
- [ ] Write base handler tests
- [ ] Document base handler API

### Phase 2: DailyTalkHandler
- [ ] Create `daily_talk_handler.py`
- [ ] Extract daily talk methods from main_system
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] Update main_system to use handler
- [ ] Verify all tests pass

### Phase 3: RestaurantHandler
- [ ] Create `restaurant_handler.py`
- [ ] Extract restaurant methods from main_system
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] Update main_system to use handler
- [ ] Verify all tests pass

### Phase 4: AttractionHandler
- [ ] Create `attraction_handler.py`
- [ ] Extract attraction methods from main_system
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] Update main_system to use handler
- [ ] Verify all tests pass

### Phase 5: TransportationHandler
- [ ] Create `transportation_handler.py`
- [ ] Extract transportation methods from main_system
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] Update main_system to use handler
- [ ] Verify all tests pass

### Phase 6: EventHandler
- [ ] Create `event_handler.py`
- [ ] Extract event methods from main_system
- [ ] Write unit tests
- [ ] Update main_system to use handler
- [ ] Verify all tests pass

### Phase 7: WeatherHandler
- [ ] Create `weather_handler.py`
- [ ] Extract weather methods from main_system
- [ ] Write unit tests
- [ ] Update main_system to use handler
- [ ] Verify all tests pass

### Phase 8: NeighborhoodHandler
- [ ] Create `neighborhood_handler.py`
- [ ] Extract neighborhood methods from main_system
- [ ] Write unit tests
- [ ] Update main_system to use handler
- [ ] Verify all tests pass

---

## 🚀 Let's Start!

**Next Step:** Create Base Handler and then start with DailyTalkHandler

**Ready to begin?** Let me know and I'll start implementing Phase 1!
