# Pure LLM Handler - Modular Architecture

**Version:** 2.0.0  
**Status:** ✅ Production Ready  
**Date:** November 2025

## 🎉 Overview

A complete modularization of the Pure LLM Handler system into 10 clean, maintainable, and testable modules. This architecture provides enterprise-grade query processing with advanced features like multi-intent detection, conversation management, A/B testing, and auto-tuning.

## 📦 Module Structure

```
backend/services/llm/
├── __init__.py                    # Module exports & version
├── core.py                        # Central orchestration (950 lines)
├── signals.py                     # Multi-intent detection (450 lines)
├── context.py                     # Smart context building (400 lines)
├── prompts.py                     # Prompt engineering (450 lines)
├── analytics.py                   # Analytics & monitoring (450 lines)
├── caching.py                     # Dual-layer caching (350 lines)
├── query_enhancement.py           # Query enhancement (450 lines)
├── conversation.py                # Conversation management (400 lines)
├── experimentation.py             # A/B testing & learning (500 lines)
├── MODULARIZATION_PROGRESS.md     # Progress tracking
└── README.md                      # This file
```

**Total:** ~4,400 lines of production-ready code

## 🚀 Quick Start

### Basic Usage

```python
from backend.services.llm import create_pure_llm_core

# Initialize
core = create_pure_llm_core(
    llm_client=your_llm_client,
    db_connection=your_db,
    config={
        'enable_semantic_cache': True,
        'enable_spell_check': True,
        'enable_ab_testing': False,
        'weather_service': weather_service,
        'events_service': events_service,
        'rag_service': rag_service
    }
)

# Process query
result = await core.process_query(
    query="Best seafood restaurants in Besiktas?",
    user_id="user123",
    language="en"
)

print(result['response'])
print(f"Signals: {result['signals']}")
print(f"Processing time: {result['metadata']['processing_time']:.2f}s")
```

### Streaming Support

```python
# Real-time streaming with progress updates
async for chunk in core.process_query_stream(
    query="How do I get to Taksim Square?",
    user_id="user123",
    language="en"
):
    if chunk['type'] == 'progress':
        print(f"Stage: {chunk['stage']}")
    elif chunk['type'] == 'token':
        print(chunk['data'], end='', flush=True)
    elif chunk['type'] == 'complete':
        print(f"\n\nProcessed in {chunk['data']['metadata']['processing_time']:.2f}s")
```

### Conversational Context

```python
# Multi-turn conversation
result = await core.process_query(
    query="Best restaurants in Sultanahmet?",
    user_id="user123",
    session_id="session_abc",
    language="en"
)

# Follow-up with reference
result = await core.process_query(
    query="How do I get there?",  # "there" will be resolved
    user_id="user123",
    session_id="session_abc",
    language="en"
)
```

## 📋 Features by Module

### 1. **core.py** - Central Orchestration
- Complete query processing pipeline
- Streaming support with progress updates
- Response validation and fallback
- Integration with all subsystems
- Helper methods for analytics, feedback, etc.

### 2. **signals.py** - Multi-Intent Detection
- 10 supported signals (restaurant, attraction, transportation, etc.)
- Keyword-based detection (fast)
- Semantic similarity detection (accurate)
- Language-aware thresholds (en, tr, ar)
- A/B testing integration

### 3. **context.py** - Smart Context Building
- Conditional database queries
- RAG integration
- Weather, events, hidden gems services
- Map generation
- Signal-based optimization

### 4. **prompts.py** - Prompt Engineering
- System prompts for en/tr
- Intent-specific instructions
- Dynamic context injection
- Conversation formatting
- Few-shot and chain-of-thought support

### 5. **analytics.py** - Analytics & Monitoring
- Query and user tracking
- Performance metrics (latency percentiles)
- Error tracking
- Signal analytics
- Hourly trends
- System health monitoring

### 6. **caching.py** - Dual-Layer Caching
- Exact match cache (fast)
- Semantic cache (finds similar queries)
- Redis + in-memory fallback
- TTL management
- LRU eviction

### 7. **query_enhancement.py** - Query Enhancement
- Spell checking (Istanbul-aware)
- Query rewriting
- Quality validation
- Autocomplete suggestions
- Trending and popular queries

### 8. **conversation.py** - Conversation Management
- Multi-turn history
- Reference resolution ("it", "there")
- Entity extraction
- Topic tracking
- Session management

### 9. **experimentation.py** - A/B Testing & Learning
- Complete A/B test framework
- Threshold learning from feedback
- Auto-tuning system
- Statistical analysis
- Winner determination

## 🎯 Architecture Benefits

### Separation of Concerns
Each module has a single responsibility:
- ✅ Easy to understand
- ✅ Easy to test
- ✅ Easy to modify
- ✅ Easy to extend

### Modularity
- ✅ Optional dependencies
- ✅ Graceful degradation
- ✅ Independent development
- ✅ Parallel testing

### Performance
- ✅ Conditional service calls
- ✅ Dual-layer caching
- ✅ Optimized context building
- ✅ Streaming support

### Monitoring
- ✅ Comprehensive analytics
- ✅ Error tracking
- ✅ Performance metrics
- ✅ System health indicators

## 📊 Query Processing Pipeline

```
User Query
    ↓
[Query Enhancement] → spell check, rewrite, validate
    ↓
[Cache Check] → semantic similarity + exact match
    ↓
[Signal Detection] → multi-intent, semantic matching
    ↓
[Conversation] → resolve references, add history
    ↓
[Context Building] → database, RAG, services
    ↓
[Prompt Engineering] → build optimized prompt
    ↓
[LLM Generation] → RunPod/OpenAI API
    ↓
[Validation] → quality checks, fallback if needed
    ↓
[Caching] → store for future queries
    ↓
[Analytics] → track metrics and performance
    ↓
Response
```

## 🔧 Configuration Options

```python
config = {
    # LLM Settings
    'max_tokens': 250,
    'temperature': 0.7,
    
    # Caching
    'enable_semantic_cache': True,
    'cache_ttl': 3600,  # seconds
    'similarity_threshold': 0.85,
    
    # Query Enhancement
    'enable_spell_check': True,
    'enable_rewriting': True,
    'enable_validation': True,
    
    # Conversation
    'max_conversation_history': 10,
    'enable_reference_resolution': True,
    
    # Analytics
    'enable_detailed_tracking': True,
    
    # Experimentation
    'enable_ab_testing': False,
    'enable_threshold_learning': True,
    'auto_tune_interval_hours': 24,
    
    # Signal Detection
    'language_thresholds': {
        'en': {'needs_restaurant': 0.30, ...},
        'tr': {'needs_restaurant': 0.35, ...}
    },
    
    # Services
    'rag_service': rag_service,
    'weather_service': weather_service,
    'events_service': events_service,
    'hidden_gems_service': hidden_gems_service,
    'map_service': map_service,
    
    # Database
    'redis_client': redis_client
}
```

## 📈 Analytics & Monitoring

### Get System Health

```python
health = core.analytics.get_system_health()
print(f"Status: {health['status']}")
print(f"Health Score: {health['health_score']:.1f}/100")
print(f"Error Rate: {health['indicators']['error_rate']:.2%}")
print(f"Avg Latency: {health['indicators']['avg_latency']:.2f}s")
print(f"Cache Hit Rate: {health['indicators']['cache_hit_rate']:.2%}")

for rec in health['recommendations']:
    print(f"- {rec}")
```

### Get Analytics Summary

```python
summary = core.get_analytics_summary()
print(f"Total Queries: {summary['basic_stats']['total_queries']}")
print(f"Unique Users: {summary['users']['unique_users']}")
print(f"Cache Hit Rate: {summary['performance']['cache_efficiency']['hit_rate']:.2%}")
print(f"Avg Query Latency: {summary['performance']['query_latency']['avg']:.2f}s")
```

## 🧪 A/B Testing

### Create Experiment

```python
# Create threshold tuning experiment
experiment = core.experimentation.create_experiment(
    name="Restaurant Signal Threshold",
    description="Test lower threshold for better recall",
    variants={
        'control': {'threshold': 0.30, 'signal': 'needs_restaurant'},
        'treatment': {'threshold': 0.25, 'signal': 'needs_restaurant'}
    },
    traffic_allocation={'control': 0.8, 'treatment': 0.2},
    success_metrics=['detection_accuracy', 'f1_score'],
    min_sample_size=100,
    auto_start=True
)
```

### Record User Feedback

```python
core.record_user_feedback(
    query="Best restaurants in Beyoglu",
    detected_signals={'needs_restaurant': True},
    confidence_scores={'needs_restaurant': 0.85},
    feedback_type='explicit',
    feedback_data={'type': 'thumbs_up'},
    language='en'
)
```

### Auto-Tune Thresholds

```python
# Run auto-tuning (usually scheduled)
results = await core.auto_tune_thresholds(language='en', force=True)
print(f"Status: {results['status']}")
for signal, rec in results['recommendations'].items():
    print(f"{signal}: {rec['current']:.3f} → {rec['recommended']:.3f}")
    print(f"  Reason: {rec['reason']}")
```

## 🔄 Migration from Old System

The new modular system is **backward compatible** with the same API:

```python
# Old system
handler = PureLLMHandler(...)
result = await handler.process_query(query, user_id, language)

# New system (same API!)
from backend.services.llm import create_pure_llm_core
core = create_pure_llm_core(...)
result = await core.process_query(query, user_id, language)
```

## 🧪 Testing

### Unit Tests

```python
# Test individual modules
from backend.services.llm.signals import SignalDetector

detector = SignalDetector()
signals = await detector.detect_signals(
    query="Best restaurants in Beyoglu",
    language="en"
)

assert signals['signals']['needs_restaurant'] == True
```

### Integration Tests

```python
# Test full pipeline
result = await core.process_query(
    query="Test query",
    user_id="test_user",
    language="en"
)

assert result['status'] == 'success'
assert 'response' in result
assert 'signals' in result
```

## 📝 Development Guidelines

### Adding a New Signal

1. **Update `signals.py`:**
   - Add patterns to `signal_patterns`
   - Add threshold to `_default_thresholds()`

2. **Update `prompts.py`:**
   - Add intent-specific prompt in `_default_intent_prompts()`

3. **Update `context.py`:**
   - Add context building logic if needed

### Adding a New Service

1. **Update `context.py`:**
   - Add service in `__init__`
   - Add context method (e.g., `_get_service_context()`)
   - Call in `build_context()`

2. **Update `core.py`:**
   - Add service to config
   - Pass to `ContextBuilder`

## 🐛 Troubleshooting

### High Latency

```python
# Check performance metrics
summary = core.get_analytics_summary()
perf = summary['performance']['query_latency']
print(f"P95 Latency: {perf['p95']:.2f}s")
print(f"P99 Latency: {perf['p99']:.2f}s")

# Check LLM latency
llm_perf = summary['performance']['llm_latency']
print(f"Avg LLM Latency: {llm_perf['avg']:.2f}s")
```

### Low Cache Hit Rate

```python
# Check cache efficiency
cache_stats = core.cache_manager.get_statistics()
print(f"Hit Rate: {cache_stats['hit_rate']:.2%}")
print(f"Exact Hits: {cache_stats['exact_hits']}")
print(f"Semantic Hits: {cache_stats['semantic_hits']}")
print(f"Misses: {cache_stats['misses']}")
```

### Signal Detection Issues

```python
# Check signal analytics
summary = core.get_analytics_summary()
signals = summary['signals']['detections_by_signal']
for signal, count in signals.items():
    print(f"{signal}: {count} detections")
```

## 📚 Documentation

- **API Reference:** See docstrings in each module
- **Architecture:** See `MODULARIZATION_PROGRESS.md`
- **Examples:** See `examples/` directory (coming soon)

## 🤝 Contributing

1. Each module should maintain a single responsibility
2. Add comprehensive docstrings
3. Include type hints
4. Add logging for important operations
5. Handle errors gracefully
6. Write unit tests

## 📄 License

Proprietary - AI Istanbul Team

## 🎉 Acknowledgments

Built with ❤️ by the AI Istanbul Team  
November 2025

---

**Status:** ✅ Production Ready  
**Version:** 2.0.0  
**Lines of Code:** ~4,400  
**Test Coverage:** TBD  
**Performance:** Optimized for <2s response time
