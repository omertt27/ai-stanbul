# Phase 4.4 Proactive Suggestions - Quick Reference

**For Developers**  
**Date**: December 3, 2025

---

## 🚀 Quick Start

### Import and Use
```python
from services.llm import (
    get_suggestion_analyzer,
    get_suggestion_generator,
    get_suggestion_presenter
)

# Initialize with LLM client
analyzer = get_suggestion_analyzer(llm_client)
generator = get_suggestion_generator(llm_client)
presenter = get_suggestion_presenter()

# Generate suggestions
context = await analyzer.analyze_context(
    query="Show me restaurants in Sultanahmet",
    response="Here are 5 great restaurants...",
    detected_intents=['find_restaurant'],
    entities={'location': 'Sultanahmet'},
    response_type='restaurant'
)

suggestions = await generator.generate_suggestions(context, max_suggestions=5)
formatted = presenter.format_for_chat(suggestions)
```

---

## 📦 Models

### SuggestionContext
```python
context = SuggestionContext(
    current_query="user query",
    current_response="bot response",
    detected_intents=["restaurant"],
    extracted_entities={"location": "Sultanahmet"},
    response_type="restaurant",
    trigger_confidence=0.85
)
```

### ProactiveSuggestion
```python
suggestion = ProactiveSuggestion(
    suggestion_id="sugg_abc123",
    suggestion_text="Get directions to these restaurants",
    suggestion_type="practical",  # exploration, practical, cultural, dining, refinement
    intent_type="get_directions",
    entities={"destination": "Sultanahmet"},
    relevance_score=0.95,
    icon="🗺️"
)
```

---

## 🔧 Configuration

### Analyzer Config
```python
config = {
    'timeout_seconds': 3,
    'temperature': 0.3,
    'max_tokens': 500,
    'min_confidence': 0.6,
    'fallback_enabled': True
}
analyzer = get_suggestion_analyzer(llm_client, config)
```

### Generator Config
```python
config = {
    'timeout_seconds': 4,
    'temperature': 0.8,
    'max_tokens': 800,
    'max_suggestions': 5,
    'min_diversity_score': 0.6,
    'fallback_enabled': True
}
generator = get_suggestion_generator(llm_client, config)
```

### Presenter Config
```python
config = {
    'default_format': 'inline',  # inline, grouped, pills, dropdown
    'show_icons': True,
    'show_categories': False,
    'max_display_suggestions': 5,
    'track_interactions': True
}
presenter = get_suggestion_presenter(config)
```

---

## 🎨 Display Formats

### Inline (Default)
```python
formatted = presenter.format_suggestions(response, 'inline')
# Returns: Display text with numbered list
```

### Grouped by Category
```python
formatted = presenter.format_suggestions(response, 'grouped')
# Returns: Groups by suggestion_type
```

### Pills/Chips
```python
formatted = presenter.format_suggestions(response, 'pills')
# Returns: Clickable pill data
```

### For Chat API
```python
formatted = presenter.format_for_chat(response)
# Returns: { 'suggestions': [...], 'metadata': {...} }
```

---

## 📊 Tracking Interactions

```python
# Track user click
interaction = presenter.track_interaction(
    suggestion_id="sugg_abc123",
    action="clicked",  # clicked, ignored, dismissed, rated
    session_id="session_123",
    query_after="Get directions to Hamdi Restaurant",
    rating=5  # Optional 1-5
)

# Get statistics
stats = presenter.get_interaction_stats()
# Returns: {
#     'total_interactions': 100,
#     'acceptance_rate': 0.35,
#     'avg_rating': 4.2
# }
```

---

## 🎯 Suggestion Types

| Type | Icon | Use Case | Examples |
|------|------|----------|----------|
| **exploration** | 🗺️💎🏛️ | Discover new places | "Discover hidden gems", "See nearby attractions" |
| **practical** | 🎯🗺️🌤️ | Useful travel info | "Get directions", "Check weather", "Find transport" |
| **cultural** | 🎭🎨🎟️ | Cultural activities | "See events tonight", "Check exhibitions" |
| **dining** | 🍽️☕🍴 | Food & restaurants | "Find restaurants nearby", "Try local cuisine" |
| **refinement** | ⚙️🔍📊 | Filter/adjust results | "Filter by price", "Show only outdoor" |

---

## 🔄 Integration Pattern

```python
# In your chat endpoint:

# 1. Get conversation context
context = await analyzer.analyze_context(
    query=user_query,
    response=bot_response,
    detected_intents=[intent],
    entities=extracted_entities,
    conversation_history=history,
    response_type=intent
)

# 2. Check if we should show suggestions
should_suggest, confidence = await analyzer.should_suggest(context)

if should_suggest:
    # 3. Generate suggestions
    suggestion_response = await generator.generate_with_response(
        context, max_suggestions=5
    )
    
    # 4. Format for display
    formatted = presenter.format_for_chat(suggestion_response)
    
    # 5. Add to response
    return ChatResponse(
        response=bot_response,
        suggestions=formatted['suggestions'],
        ...
    )
```

---

## 🧪 Testing

### Unit Test Example
```python
import pytest
from services.llm import get_suggestion_generator

@pytest.mark.asyncio
async def test_suggestion_generation():
    # Mock LLM client
    class MockLLM:
        async def chat_completion(self, **kwargs):
            return {
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "suggestions": [
                                {
                                    "text": "Get directions",
                                    "type": "practical",
                                    "intent": "get_directions",
                                    "relevance": 0.95
                                }
                            ]
                        })
                    }
                }]
            }
    
    generator = get_suggestion_generator(MockLLM())
    
    context = SuggestionContext(
        current_query="restaurants in Sultanahmet",
        current_response="Here are 5 restaurants...",
        response_type="restaurant"
    )
    
    suggestions = await generator.generate_suggestions(context)
    assert len(suggestions) > 0
    assert suggestions[0].suggestion_type == "practical"
```

---

## 📈 Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Generation Time | <100ms | Total time for all 3 components |
| Relevance Score | >0.85 | LLM-assigned relevance |
| Diversity Score | >0.6 | Variety across types |
| Acceptance Rate | >30% | Users clicking suggestions |
| LLM Usage | 90% | Primary decision maker |

---

## 🐛 Common Issues

### Issue: No suggestions generated
**Solution**: Check trigger confidence >= 0.6, ensure LLM client available

### Issue: Suggestions irrelevant
**Solution**: Tune prompts, verify entity extraction, check context

### Issue: Slow generation
**Solution**: Reduce temperature, enable caching, use template fallback

### Issue: Same suggestions every time
**Solution**: Check diversity scoring, increase temperature, review templates

---

## 📝 Template Suggestions

Default templates by response type (fallback when LLM unavailable):

```python
'restaurant': [
    'Get directions to these restaurants',
    'Check the weather forecast',
    'Find attractions nearby',
    'Discover hidden gems in {location}',
    'See cultural events tonight'
],
'attraction': [
    'Get directions to this attraction',
    'Find restaurants nearby',
    'See other attractions in the area',
    'Check opening hours and prices',
    'Discover hidden gems nearby'
],
# ... more in suggestion_generator.py
```

---

## 🔍 Debugging

### Enable Debug Logging
```python
import logging
logging.getLogger('services.llm.suggestion_analyzer').setLevel(logging.DEBUG)
logging.getLogger('services.llm.suggestion_generator').setLevel(logging.DEBUG)
```

### Check Component Status
```python
from api.chat import get_suggestion_components

analyzer, generator, presenter = get_suggestion_components()
print(f"Analyzer: {analyzer is not None}")
print(f"Generator: {generator is not None}")
print(f"Presenter: {presenter is not None}")
```

### Inspect Suggestion Response
```python
suggestion_response = await generator.generate_with_response(context)
print(f"Method: {suggestion_response.generation_method}")
print(f"Time: {suggestion_response.generation_time_ms}ms")
print(f"Confidence: {suggestion_response.confidence}")
print(f"Diversity: {suggestion_response.diversity_score}")
print(f"LLM Used: {suggestion_response.llm_used}")
```

---

## 🎯 Best Practices

1. **Always provide context**: More context = better suggestions
2. **Handle errors gracefully**: Always have fallback suggestions
3. **Track interactions**: Use data to improve prompts
4. **Test with real queries**: LLM behavior varies
5. **Monitor performance**: Track generation time
6. **A/B test**: Try different counts and formats
7. **Cache common patterns**: Speed up frequent scenarios
8. **Tune confidence thresholds**: Balance quality vs quantity

---

## 📚 Related Documentation

- `PHASE4_4_PROACTIVE_SUGGESTIONS_PLAN.md` - Full planning document
- `PHASE4_4_INTEGRATION_COMPLETE.md` - Integration guide
- `backend/services/llm/suggestion_analyzer.py` - Analyzer implementation
- `backend/services/llm/suggestion_generator.py` - Generator implementation
- `backend/services/llm/suggestion_presenter.py` - Presenter implementation
- `backend/api/chat.py` - Integration in chat endpoint

---

## 💡 Pro Tips

- Use higher temperature (0.8) for more creative suggestions
- Lower confidence threshold (0.5) to show more suggestions
- Group format works best for 4+ suggestions
- Pills format is mobile-friendly
- Track which suggestions get clicked to improve prompts
- Cache suggestions for identical queries
- Use entities to pre-fill suggestion parameters

---

**Quick Reference Version**: 1.0  
**Last Updated**: December 3, 2025  
**Status**: Production Ready ✅
