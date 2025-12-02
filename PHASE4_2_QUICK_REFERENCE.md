# Phase 4.2 Quick Reference Guide

**LLM Conversation Context Manager - Developer Guide**

---

## 🚀 Quick Start

### Basic Usage

```python
from services.llm import get_context_manager

# Get singleton instance
context_manager = get_context_manager(
    llm_client=your_llm_client,
    config={'enable_llm': True}
)

# Resolve context
result = await context_manager.resolve_context(
    current_query="What about restaurants there?",
    session_id="user_123_session",
    user_id="user_123",
    user_location={"lat": 41.0082, "lon": 28.9784}
)

# Check result
if result['needs_clarification']:
    return result['clarification_question']
else:
    use_query = result['resolved_query']
```

---

## 📦 Import Options

### Option 1: Get Singleton
```python
from services.llm import get_context_manager

manager = get_context_manager(llm_client=client)
result = await manager.resolve_context(query, session_id)
```

### Option 2: Direct Function
```python
from services.llm import resolve_conversation_context

result = await resolve_conversation_context(
    current_query=query,
    session_id=session_id,
    llm_client=client
)
```

### Option 3: Create Instance
```python
from services.llm import LLMConversationContextManager

manager = LLMConversationContextManager(
    llm_client=client,
    config={'max_history_turns': 5}
)
```

---

## 🔧 Configuration

### Available Options

```python
config = {
    'enable_llm': True,           # Use LLM for resolution (default: True)
    'fallback_to_rules': True,    # Fallback if LLM fails (default: True)
    'timeout_seconds': 2,         # LLM timeout (default: 2)
    'max_history_turns': 10,      # Max history to consider (default: 10)
    'min_confidence': 0.6         # Min confidence threshold (default: 0.6)
}

manager = get_context_manager(llm_client=client, config=config)
```

---

## 📊 Response Format

### Full Response Structure

```python
{
    # Resolution status
    'has_references': bool,          # True if query has pronouns/references
    'source': 'llm' | 'fallback',    # Resolution method used
    'confidence': float,             # 0.0-1.0
    
    # Resolved content
    'resolved_query': str,           # Standalone query with context
    'resolved_references': {         # What was resolved
        'there': 'Hagia Sophia',
        'it': 'Blue Mosque'
    },
    
    # Inferred context
    'implicit_context': {
        'origin': 'Taksim Square',
        'destination': 'Galata Tower',
        'continuing_task': 'route_planning',
        'user_intent': 'get_directions',
        'topic': 'navigation'
    },
    
    # Clarification
    'needs_clarification': bool,     # True if ambiguous
    'clarification_question': str,   # Question to ask user
    'missing_information': ['destination', ...],
    
    # Metadata
    'reasoning': str,                # LLM's analysis
    'session_state': {               # Current session info
        'session_id': str,
        'last_mentioned_locations': [],
        'history_length': int,
        ...
    }
}
```

---

## 🎯 Common Patterns

### Pattern 1: Check for Clarification

```python
result = await manager.resolve_context(query, session_id)

if result['needs_clarification']:
    # Ask user for clarification
    return {
        'response': result['clarification_question'],
        'intent': 'clarification',
        'confidence': 0.8
    }

# Continue with resolved query
resolved_query = result['resolved_query']
```

### Pattern 2: Extract Implicit Context

```python
result = await manager.resolve_context(query, session_id)

implicit = result.get('implicit_context', {})

origin = implicit.get('origin')
destination = implicit.get('destination')
intent = implicit.get('user_intent')

# Use for routing
if origin and destination:
    route = calculate_route(origin, destination)
```

### Pattern 3: Check Confidence

```python
result = await manager.resolve_context(query, session_id)

if result['confidence'] < 0.6:
    # Low confidence - ask for confirmation
    return f"Did you mean: {result['resolved_query']}?"

# High confidence - proceed
process_query(result['resolved_query'])
```

### Pattern 4: Track References

```python
result = await manager.resolve_context(query, session_id)

if result['has_references']:
    refs = result['resolved_references']
    logger.info(f"Resolved references: {refs}")
    
    # Show user what was resolved
    if refs:
        context_hint = f"(referring to {', '.join(refs.values())})"
```

---

## 💾 Session Management

### Create/Get Session

```python
# Automatically creates if doesn't exist
session = manager.get_or_create_session(
    session_id="user_123_session",
    user_id="user_123"
)

print(f"Session has {len(session.conversation_history)} turns")
```

### Record Conversation Turn

```python
manager.record_turn(
    session_id="user_123_session",
    user_query="Show me route to Hagia Sophia",
    bot_response="Here's your route...",
    intent="route",
    locations=["Hagia Sophia"],
    entities={'origin': 'Taksim', 'destination': 'Hagia Sophia'}
)
```

### Clear Session

```python
# Clear old session
manager.clear_session("old_session_id")
```

### Get Statistics

```python
stats = manager.get_stats()

print(f"Total resolutions: {stats['total_resolutions']}")
print(f"LLM usage rate: {stats['llm_usage_rate']:.1%}")
print(f"Active sessions: {stats['active_sessions']}")
print(f"Average latency: {stats['average_latency_ms']:.0f}ms")
```

---

## 🧪 Testing

### Mock LLM Client

```python
from unittest.mock import Mock, MagicMock

# Create mock client
mock_client = MagicMock()
mock_client.chat.completions.create = Mock(
    return_value=create_mock_response('''
    {
        "has_references": true,
        "resolved_references": {"there": "Hagia Sophia"},
        "resolved_query": "What restaurants are near Hagia Sophia?",
        "confidence": 0.95
    }
    ''')
)

# Use in tests
manager = LLMConversationContextManager(llm_client=mock_client)
result = await manager.resolve_context(query, session_id)
```

### Test Fallback

```python
# Test with no LLM client
manager = LLMConversationContextManager(
    llm_client=None,
    config={'enable_llm': False, 'fallback_to_rules': True}
)

result = await manager.resolve_context(query, session_id)
assert result['source'] == 'fallback'
```

---

## 🚨 Error Handling

### Graceful Degradation

```python
try:
    result = await manager.resolve_context(query, session_id)
    
    if result['source'] == 'fallback':
        logger.warning("Using fallback context resolution")
    
    # Use result either way
    process_query(result['resolved_query'])
    
except Exception as e:
    logger.error(f"Context resolution failed: {e}")
    # Continue without context resolution
    process_query(original_query)
```

### Timeout Handling

```python
# Configure timeout
manager = get_context_manager(
    llm_client=client,
    config={'timeout_seconds': 3}  # 3 second timeout
)

# Fallback will be used if timeout
result = await manager.resolve_context(query, session_id)
```

---

## 📈 Performance Tips

### 1. Use Singleton Pattern

```python
# ✅ Good - reuses instance
manager = get_context_manager(llm_client)

# ❌ Bad - creates new instance each time
manager = LLMConversationContextManager(llm_client)
```

### 2. Limit History Size

```python
# For faster processing
config = {'max_history_turns': 5}  # Only last 5 turns
manager = get_context_manager(llm_client, config)
```

### 3. Monitor Stats

```python
stats = manager.get_stats()

if stats['llm_usage_rate'] < 0.8:
    logger.warning("LLM usage rate is low - check for errors")

if stats['average_latency_ms'] > 500:
    logger.warning("High latency - consider reducing max_history_turns")
```

---

## 🔍 Debugging

### Enable Debug Logging

```python
import logging

logging.getLogger('conversation_context_manager').setLevel(logging.DEBUG)
```

### Inspect Session State

```python
result = await manager.resolve_context(query, session_id)

# Get full session state
session_state = result['session_state']

print(f"Session: {session_state['session_id']}")
print(f"History: {session_state['history_length']} turns")
print(f"Locations: {session_state['last_mentioned_locations']}")
print(f"Preferences: {session_state['user_preferences']}")
```

### Check LLM Reasoning

```python
result = await manager.resolve_context(query, session_id)

if 'reasoning' in result:
    print(f"LLM reasoning: {result['reasoning']}")
```

---

## 📚 Example Scenarios

### Scenario 1: Restaurant Search with Context

```python
# Turn 1: User mentions location
manager.record_turn(
    session_id="user_123",
    user_query="I'm at Hagia Sophia",
    bot_response="Great! How can I help?",
    locations=["Hagia Sophia"]
)

# Turn 2: User asks about restaurants
result = await manager.resolve_context(
    current_query="What about restaurants nearby?",
    session_id="user_123"
)

# Result:
# resolved_query: "What restaurants are near Hagia Sophia?"
# implicit_context: {'destination': 'Hagia Sophia'}
```

### Scenario 2: Multi-Stop Journey

```python
# Turn 1: Route request
manager.record_turn(
    session_id="user_456",
    user_query="Show me route from Taksim to Galata Tower",
    bot_response="Here's your route...",
    intent="route",
    locations=["Taksim", "Galata Tower"]
)

# Turn 2: Add stop
result = await manager.resolve_context(
    current_query="Can we stop at Istiklal Avenue?",
    session_id="user_456"
)

# Result:
# resolved_query: "Add Istiklal Avenue as a stop on route from Taksim to Galata Tower"
# implicit_context: {'continuing_task': 'route_planning'}
```

---

## ✅ Best Practices

1. **Always record turns** after generating responses
2. **Check clarification** before processing queries
3. **Use resolved query** for downstream processing
4. **Monitor statistics** to ensure LLM usage
5. **Handle errors gracefully** with fallback
6. **Keep session IDs consistent** per user
7. **Clear old sessions** to prevent memory buildup
8. **Log LLM reasoning** for debugging

---

## 🆘 Troubleshooting

### Issue: Low Confidence Results

```python
# Increase history size
config = {'max_history_turns': 10}

# Or record more turn details
manager.record_turn(..., entities={...}, locations=[...])
```

### Issue: Slow Performance

```python
# Reduce history size
config = {'max_history_turns': 3, 'timeout_seconds': 1}
```

### Issue: Fallback Used Too Often

```python
# Check stats
stats = manager.get_stats()
print(f"Fallback rate: {stats['fallback_resolutions']/stats['total_resolutions']:.1%}")

# If high, check LLM client configuration
```

---

## 📞 Support

- **Documentation**: `PHASE4_2_CONVERSATION_CONTEXT_COMPLETE.md`
- **Tests**: `test_phase4_2_context_integration.py`
- **Source**: `backend/services/llm/conversation_context_manager.py`

---

**Quick Reference Version**: 1.0  
**Last Updated**: December 2, 2025
