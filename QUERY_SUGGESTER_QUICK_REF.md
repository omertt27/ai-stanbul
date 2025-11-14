# QuerySuggester Quick Reference

**One-page developer cheat sheet for Priority 4.1**

---

## Installation

```bash
pip install fuzzywuzzy python-Levenshtein
```

---

## Basic Setup

```python
from backend.services.query_suggester import create_query_suggester

# Initialize
suggester = create_query_suggester(
    llm_client=your_llm_client,
    redis_client=your_redis_client  # Optional
)
```

---

## API Reference

### 1️⃣ Autocomplete (<10ms)

```python
suggestions = await suggester.suggest_completions(
    partial_query="best resta",
    language="en",
    max_suggestions=5
)
# Returns: ["best restaurants in Taksim", "best restaurants Sultanahmet", ...]
```

### 2️⃣ Spell Correction (~15ms)

```python
correction = await suggester.suggest_correction(
    query="hotels in Taksin",  # Typo
    language="en",
    threshold=0.75  # Similarity threshold
)
# Returns: {
#   "corrected_query": "hotels in Taksim",
#   "confidence": 0.92,
#   "changes": [{"original": "Taksin", "corrected": "Taksim", ...}]
# }
```

### 3️⃣ Related Queries (~500ms first, ~2ms cached)

```python
related = await suggester.suggest_related(
    query="best museums in Istanbul",
    response="The best museums are...",
    signals={"primary_intent": "tourism"},
    language="en",
    max_suggestions=3
)
# Returns: [
#   "What are the museum opening hours?",
#   "How much do museum tickets cost?",
#   "Which museums are free?"
# ]
```

### 4️⃣ Track Query (non-blocking)

```python
suggester.track_query("best restaurants in Taksim")
# Updates popularity for autocomplete
```

### 5️⃣ Get Statistics

```python
stats = suggester.get_stats()
# Returns: {
#   "autocomplete": {"requests": 1523, ...},
#   "spell_check": {"corrections_made": 89, ...},
#   "related_queries": {"cache_hit_rate": 0.73, ...}
# }
```

---

## Common Patterns

### Pattern 1: Full Query Processing

```python
async def process_query(user_input, language="en"):
    # 1. Spell check
    correction = await suggester.suggest_correction(user_input, language)
    query = correction['corrected_query'] if correction else user_input
    
    # 2. Track for popularity
    suggester.track_query(query)
    
    # 3. Process with LLM
    response, signals = await llm.process(query)
    
    # 4. Generate related queries
    related = await suggester.suggest_related(query, response, signals, language)
    
    return {
        "response": response,
        "corrected_from": user_input if correction else None,
        "related_queries": related
    }
```

### Pattern 2: Autocomplete Endpoint

```python
@app.post("/api/autocomplete")
async def autocomplete(partial: str, language: str = "en"):
    suggestions = await suggester.suggest_completions(partial, language, max_suggestions=5)
    return {"suggestions": suggestions}
```

### Pattern 3: Smart Correction (only high confidence)

```python
correction = await suggester.suggest_correction(query)
if correction and correction['confidence'] > 0.90:
    # Auto-apply correction
    query = correction['corrected_query']
else:
    # Show suggestion to user
    # "Did you mean: {corrected_query}?"
    pass
```

---

## Configuration

### Custom Locations

```python
suggester = QuerySuggester(
    llm_client=llm_client,
    redis_client=redis_client,
    location_names=[
        "Custom Location 1",
        "Custom Location 2",
        # ... your locations
    ]
)
```

### Thresholds

```python
# Spell correction
correction = await suggester.suggest_correction(
    query="Taksin",
    threshold=0.70  # Lower = more lenient (50-95%)
)
```

---

## Redis Schema

```
Key                                    Type    TTL
query_suggester:popular_queries        Hash    24h
related:{md5_hash}:{language}          String  1h
query_suggester:stats                  Hash    -
```

---

## Performance

| Feature          | Latency  | Notes                    |
|------------------|----------|--------------------------|
| Autocomplete     | <10ms    | In-memory trie           |
| Spell Correction | ~15ms    | Fuzzy matching           |
| Related (cached) | ~2ms     | Redis lookup             |
| Related (uncached)| ~500ms  | LLM call                 |
| Query Tracking   | <1ms     | Fire-and-forget          |

---

## Supported Languages

- **English** (en) ✅
- **Turkish** (tr) ✅
- **Arabic** (ar) ✅
- **Spanish** (es) ✅
- **French** (fr) ✅
- **German** (de) ✅

---

## Error Handling

All methods handle errors gracefully:

```python
# Autocomplete - returns empty list on error
suggestions = await suggester.suggest_completions("query")
# Never throws, always returns List[str]

# Spell correction - returns None on error
correction = await suggester.suggest_correction("query")
# Returns Optional[Dict]

# Related queries - returns empty list on LLM failure
related = await suggester.suggest_related(...)
# Never throws, always returns List[str]
```

---

## Monitoring

```python
# Get real-time stats
stats = suggester.get_stats()

# Check cache efficiency
cache_hit_rate = stats['related_queries']['cache_hit_rate']
if cache_hit_rate < 0.70:
    logger.warning("Low cache hit rate!")

# Get popular queries
popular = suggester.get_popular_queries(limit=10)
for query, frequency in popular:
    print(f"{query}: {frequency} requests")
```

---

## Testing

```bash
# Run all tests
pytest backend/test_query_suggester.py -v

# Run specific test
pytest backend/test_query_suggester.py::test_autocomplete_suggestions -v

# Check coverage
pytest backend/test_query_suggester.py --cov=backend/services/query_suggester
```

---

## Debug Mode

```python
import logging
logging.getLogger('backend.services.query_suggester').setLevel(logging.DEBUG)

# Shows:
# - Autocomplete results
# - Spell corrections made
# - Cache hits/misses
# - Related query generation
```

---

## Common Issues

### ❌ ModuleNotFoundError: No module named 'fuzzywuzzy'
```bash
pip install fuzzywuzzy python-Levenshtein
```

### ❌ Redis connection failed
```python
# QuerySuggester works without Redis (degraded mode)
suggester = create_query_suggester(llm_client, redis_client=None)
# Autocomplete and spell check still work, related queries not cached
```

### ❌ Autocomplete returning no results
```python
# Ensure queries are tracked
suggester.track_query("your query")

# Or load popular queries from Redis
suggester._load_popular_queries()
```

---

## Best Practices

✅ **DO**:
- Track all user queries for autocomplete
- Cache related queries (1-hour TTL)
- Use confidence threshold for auto-correction
- Monitor cache hit rates

❌ **DON'T**:
- Don't block on query tracking (fire-and-forget)
- Don't correct queries with low confidence (<75%)
- Don't cache autocomplete (use in-memory trie)
- Don't generate related queries without context

---

## Links

- **Full Documentation**: `PRIORITY_4_1_COMPLETE.md`
- **Integration Guide**: `PRIORITY_4_1_INTEGRATION_GUIDE.md`
- **Test Suite**: `backend/test_query_suggester.py`
- **Source Code**: `backend/services/query_suggester.py`

---

**Version**: 1.0.0  
**Status**: ✅ Production Ready  
**Tests**: 23/23 passing
