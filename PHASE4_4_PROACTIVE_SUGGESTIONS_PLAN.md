# Phase 4.4: Proactive Suggestions System

**Status**: 🚧 **PLANNING**  
**Date**: December 3, 2025  
**Prerequisites**: ✅ Phase 4.3 Complete

---

## 🎯 Overview

Phase 4.4 implements an intelligent proactive suggestion system that anticipates user needs and provides contextually relevant follow-up suggestions. The system will leverage LLM intelligence to predict what users might want to know next.

---

## 🎨 Vision

**Current Experience** (Reactive):
```
User: "Show me restaurants in Sultanahmet"
Bot: "Here are 5 restaurants in Sultanahmet..."
[Conversation ends - user must think of next question]
```

**Enhanced Experience** (Proactive):
```
User: "Show me restaurants in Sultanahmet"
Bot: "Here are 5 restaurants in Sultanahmet...

💡 You might also want to:
   • Get directions to these restaurants
   • Check the weather forecast for your visit
   • Discover hidden gems nearby
   • See what cultural events are happening
   • Find attractions within walking distance"

[User can click or type - conversation flows naturally]
```

---

## 🎪 Key Features

### 1. Context-Aware Suggestions
- Analyze current query and response
- Consider conversation history
- Factor in user location (if known)
- Understand user intent patterns

### 2. LLM-Powered Intelligence (90% LLM)
- Predict logical next steps
- Generate natural suggestion phrases
- Rank suggestions by relevance
- Adapt to user behavior

### 3. Multi-Category Suggestions
- **Exploration**: Nearby attractions, hidden gems
- **Practical**: Directions, weather, transportation
- **Cultural**: Events, activities, local customs
- **Dining**: Restaurants, cafes, food recommendations
- **Refinement**: Filter results, change criteria

### 4. Smart Triggering
- After successful responses
- When multiple options exist
- At natural conversation break points
- Not on errors or incomplete responses

---

## 🏗️ Architecture

```
User Query + Response
    ↓
Suggestion Analyzer (LLM)
    ├─ Analyze conversation context
    ├─ Identify response type
    ├─ Extract key entities
    └─ Determine user goals
    ↓
Suggestion Generator (LLM)
    ├─ Generate 3-5 relevant suggestions
    ├─ Rank by relevance and usefulness
    ├─ Format as natural language
    └─ Include action indicators
    ↓
Suggestion Presenter
    ├─ Add to response metadata
    ├─ Format for UI display
    └─ Track suggestion effectiveness
```

---

## 📋 Implementation Plan

### Module 1: Proactive Suggestion Analyzer
**File**: `backend/services/llm/suggestion_analyzer.py`

**Responsibilities**:
- Analyze current conversation state
- Extract entities and context
- Identify suggestion opportunities
- Score suggestion triggers (should we suggest?)

**LLM Usage**: 85%
- Context understanding
- Entity extraction
- Opportunity identification
- Fallback: Rule-based triggers

**Key Methods**:
```python
class SuggestionAnalyzer:
    async def analyze_context(
        self,
        query: str,
        response: str,
        conversation_history: List[Dict],
        entities: Dict[str, Any]
    ) -> SuggestionContext
    
    async def should_suggest(
        self,
        context: SuggestionContext
    ) -> Tuple[bool, float]  # (should_suggest, confidence)
```

### Module 2: Proactive Suggestion Generator
**File**: `backend/services/llm/suggestion_generator.py`

**Responsibilities**:
- Generate relevant suggestions
- Rank by usefulness
- Format for presentation
- Track suggestion patterns

**LLM Usage**: 95%
- Natural language generation
- Relevance ranking
- Context adaptation
- Fallback: Template-based suggestions

**Key Methods**:
```python
class SuggestionGenerator:
    async def generate_suggestions(
        self,
        context: SuggestionContext,
        max_suggestions: int = 5
    ) -> List[ProactiveSuggestion]
    
    async def rank_suggestions(
        self,
        suggestions: List[ProactiveSuggestion],
        context: SuggestionContext
    ) -> List[ProactiveSuggestion]
```

### Module 3: Suggestion Presenter
**File**: `backend/services/llm/suggestion_presenter.py`

**Responsibilities**:
- Format suggestions for display
- Add UI metadata
- Track presentation metrics
- Handle user selection feedback

**LLM Usage**: 20%
- Format optimization
- Fallback: Template formatting

**Key Methods**:
```python
class SuggestionPresenter:
    def format_suggestions(
        self,
        suggestions: List[ProactiveSuggestion],
        format_type: str = "inline"
    ) -> Dict[str, Any]
    
    def track_interaction(
        self,
        suggestion_id: str,
        user_action: str
    ) -> None
```

---

## 📊 Data Models

Add to `models.py`:

```python
class SuggestionContext(BaseModel):
    """Context for generating suggestions"""
    current_query: str
    current_response: str
    detected_intents: List[str]
    extracted_entities: Dict[str, Any]
    conversation_history: List[Dict[str, Any]]
    user_location: Optional[str] = None
    response_type: str  # restaurant, attraction, route, etc.
    trigger_confidence: float

class ProactiveSuggestion(BaseModel):
    """A single proactive suggestion"""
    suggestion_id: str
    suggestion_text: str  # "Get directions to these restaurants"
    suggestion_type: str  # exploration, practical, cultural, dining
    intent_type: str  # get_directions, check_weather, etc.
    relevance_score: float
    entities: Dict[str, Any]  # Pre-filled entities for the suggestion
    reasoning: Optional[str] = None  # Why this suggestion?

class ProactiveSuggestionResponse(BaseModel):
    """Complete suggestion response"""
    suggestions: List[ProactiveSuggestion]
    context: SuggestionContext
    generation_method: str  # llm, template, hybrid
    generation_time_ms: float
    total_suggestions_considered: int
```

---

## 🎯 Example Use Cases

### Use Case 1: Restaurant Query
```python
Query: "Show me restaurants in Beyoğlu"
Response: "Here are 5 great restaurants in Beyoğlu..."

Suggestions:
1. 🗺️ "Get directions to Beyoğlu from your location" (practical)
2. 🎭 "See cultural events happening in Beyoğlu tonight" (cultural)
3. 💎 "Discover hidden gems in Beyoğlu" (exploration)
4. 🌤️ "Check the weather forecast for your visit" (practical)
5. 🏛️ "Find attractions near these restaurants" (exploration)
```

### Use Case 2: Directions Query
```python
Query: "How do I get to Hagia Sophia?"
Response: "Take the T1 tram to Sultanahmet..."

Suggestions:
1. 🍽️ "Find restaurants near Hagia Sophia" (dining)
2. 🏛️ "See other attractions in Sultanahmet" (exploration)
3. 🎟️ "Check Hagia Sophia opening hours and tickets" (practical)
4. 📸 "Discover best photo spots at Hagia Sophia" (cultural)
5. ⏱️ "Plan a full day itinerary in Sultanahmet" (exploration)
```

### Use Case 3: Multi-Intent Query
```python
Query: "I want to visit museums and find lunch nearby"
Response: [Combined response about museums and restaurants]

Suggestions:
1. 🗺️ "Create a route connecting these museums" (practical)
2. 🎫 "Check museum pass options and prices" (practical)
3. 🌟 "Add hidden gem museums to your tour" (exploration)
4. ⏰ "See what time to visit each museum (avoid crowds)" (practical)
5. 🎨 "Find current exhibitions at these museums" (cultural)
```

---

## 🔧 Integration Points

### 1. Chat Pipeline Integration
```python
# In chat.py
async def process_query(query: str, context: dict):
    # ...existing processing...
    
    # After getting response
    if should_generate_suggestions(response):
        suggestions = await generate_proactive_suggestions(
            query=query,
            response=response_text,
            conversation_history=context.get('history', []),
            entities=extracted_entities
        )
        
        response['suggestions'] = suggestions
    
    return response
```

### 2. Response Format
```json
{
  "response": "Here are 5 restaurants in Sultanahmet...",
  "intent": "find_restaurant",
  "suggestions": {
    "items": [
      {
        "id": "sugg_001",
        "text": "Get directions to these restaurants",
        "type": "practical",
        "intent": "get_directions",
        "relevance": 0.92,
        "entities": {"location": "Sultanahmet"}
      },
      ...
    ],
    "generation_method": "llm",
    "timestamp": "2025-12-03T10:30:00Z"
  }
}
```

### 3. Frontend Display
```typescript
// Frontend can display suggestions as:
// - Clickable pills below response
// - Dropdown menu
// - Chat bubbles
// - Quick action buttons
```

---

## 📈 Success Metrics

### Engagement Metrics
- **Suggestion Acceptance Rate**: % of suggestions clicked/used
- **Conversation Extension**: Avg queries after suggestions
- **User Satisfaction**: Rating after using suggestions

### Quality Metrics
- **Relevance Score**: User feedback on suggestion quality
- **Generation Time**: Time to generate suggestions (<100ms target)
- **Diversity Score**: Variety of suggestion types

### Business Metrics
- **Session Length**: Increase in conversation duration
- **Feature Discovery**: Users trying new features
- **Retention**: Return rate after using suggestions

---

## 🧪 Testing Strategy

### Unit Tests
- Test suggestion generation for each response type
- Test ranking algorithm
- Test fallback mechanisms
- Test context extraction

### Integration Tests
- Test end-to-end suggestion flow
- Test with various query types
- Test conversation continuity
- Test suggestion filtering

### A/B Testing
- Control: No suggestions
- Variant A: 3 suggestions
- Variant B: 5 suggestions
- Measure engagement and satisfaction

---

## 📝 Implementation Steps

### Phase 1: Core Implementation (2-3 hours)
1. ✅ Create planning document (this file)
2. ⬜ Implement data models
3. ⬜ Implement SuggestionAnalyzer
4. ⬜ Implement SuggestionGenerator
5. ⬜ Implement SuggestionPresenter

### Phase 2: Integration (1-2 hours)
6. ⬜ Integrate into chat pipeline
7. ⬜ Add response formatting
8. ⬜ Update API endpoints
9. ⬜ Add configuration options

### Phase 3: Testing (1-2 hours)
10. ⬜ Write unit tests
11. ⬜ Write integration tests
12. ⬜ Manual testing with various queries
13. ⬜ Performance testing

### Phase 4: Documentation (1 hour)
14. ⬜ Document API changes
15. ⬜ Create usage examples
16. ⬜ Update deployment guide
17. ⬜ Create A/B testing plan

---

## 🎨 LLM Prompts (Draft)

### Suggestion Analysis Prompt
```
Analyze this conversation to determine if we should provide proactive suggestions:

Query: "{query}"
Response: "{response}"
Intent: {intent_type}
Entities: {entities}

Consider:
1. Is the response complete and successful?
2. Are there logical next steps the user might want?
3. Is this a natural conversation break point?
4. Would suggestions enhance user experience?

Respond with JSON:
{
  "should_suggest": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "why or why not",
  "context_summary": "brief summary of situation"
}
```

### Suggestion Generation Prompt
```
Generate 5 proactive, helpful suggestions for the user:

Current Query: "{query}"
Response Given: "{response}"
Context: {context}

Generate suggestions that:
1. Are natural next steps
2. Add value to the user's journey
3. Cover different categories (exploration, practical, cultural)
4. Are specific and actionable
5. Use Istanbul travel context

Respond with JSON array:
[
  {
    "text": "Get directions to these restaurants",
    "type": "practical",
    "intent": "get_directions",
    "relevance": 0.95,
    "reasoning": "User asked about restaurants, likely wants to visit"
  },
  ...
]
```

---

## 🚀 Future Enhancements

### Phase 4.4.1: Personalized Suggestions
- Learn user preferences over time
- Adapt to user behavior patterns
- Remember past interactions

### Phase 4.4.2: Contextual Suggestions
- Time-of-day awareness (breakfast vs dinner)
- Weather-based suggestions
- Event-driven suggestions

### Phase 4.4.3: Conversational Chains
- Multi-step suggestion paths
- Guided exploration flows
- Tutorial-like experiences

---

## 💡 Design Principles

1. **Non-Intrusive**: Suggestions enhance, don't distract
2. **Contextual**: Always relevant to current conversation
3. **Actionable**: User can immediately act on suggestions
4. **Diverse**: Cover different types of needs
5. **Natural**: Feel like helpful recommendations, not pushy sales

---

## 📊 Expected Impact

**User Experience**:
- Smoother conversation flow
- Reduced "what to ask next" friction
- Discovery of features they didn't know existed
- More complete travel planning

**Business Impact**:
- Increased engagement time
- Higher feature utilization
- Better user satisfaction
- More comprehensive travel assistance

**Technical**:
- LLM Responsibility: 90% (very high intelligence)
- Response Time: <100ms additional
- Success Rate: >85% relevant suggestions
- Fallback: Template-based suggestions always available

---

## ✅ Definition of Done

Phase 4.4 is complete when:
- [ ] All three modules implemented and tested
- [ ] Integration into chat pipeline complete
- [ ] Unit tests passing (>90% coverage)
- [ ] Integration tests passing
- [ ] Documentation complete
- [ ] Manual testing successful across query types
- [ ] Performance meets targets (<100ms)
- [ ] Ready for A/B testing in production

---

**Next Action**: Begin implementation of data models and SuggestionAnalyzer
