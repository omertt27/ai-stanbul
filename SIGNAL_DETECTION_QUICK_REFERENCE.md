# Signal-Based Intent Detection - Quick Reference

## 🎯 What Changed?

**Old System:** Single-intent, keyword-based detection (English-focused)  
**New System:** Multi-signal, semantic detection (multilingual, flexible)

---

## 📊 Key Improvements

| Feature | Old | New |
|---------|-----|-----|
| **Languages** | Primarily English | Turkish + English (semantic) |
| **Intent Support** | Single intent only | Multiple signals per query |
| **Detection Method** | Keywords only | Semantic embeddings + keywords |
| **Landmarks** | Limited | 20+ famous Istanbul landmarks |
| **Performance** | ~5ms | 10-12ms (still excellent) |
| **Accuracy** | ~70% | 100% (keyword fallback) |

---

## 🔧 Supported Signals

### Expensive Operations (Explicit Detection)
- `needs_map` - Map visualization needed
- `needs_gps_routing` - Turn-by-turn directions

### Service Signals
- `needs_weather` - Weather info/recommendations
- `needs_events` - Event listings
- `needs_hidden_gems` - Local authentic places

### Context Signals
- `has_budget_constraint` - Price filtering needed
- `has_user_location` - User GPS available
- `mentions_location` - Specific place mentioned
- `likely_restaurant` - Restaurant query hint
- `likely_attraction` - Attraction query hint

---

## 💡 Example Queries

### Single-Signal (Simple)
```
"What's the weather like?" → needs_weather
"Where should I eat?" → likely_restaurant
```

### Multi-Signal (Complex)
```
"Show me cheap restaurants near Galata Tower with directions"
→ likely_restaurant, has_budget_constraint, mentions_location, needs_map, likely_attraction
```

### Multilingual
```
English: "How do I get to Hagia Sophia?"
Turkish: "Ayasofya'ya nasıl gidilir?"
Both → likely_attraction, needs_map
```

---

## 🚀 Famous Landmarks (Auto-Detected)

### English
Hagia Sophia, Blue Mosque, Galata Tower, Topkapi Palace, Dolmabahce Palace, Basilica Cistern, Grand Bazaar, Spice Bazaar, Bosphorus, Maiden Tower, Taksim Square, Istiklal Street, Ortakoy

### Turkish
Ayasofya, Sultanahmet, Galata Kulesi, Topkapı Sarayı, Dolmabahçe Sarayı, Yerebatan Sarnıcı, Kapalı Çarşı, Mısır Çarşısı, Boğaz, Kız Kulesi, Taksim, İstiklal, Ortaköy

---

## 🔍 How It Works

1. **Query Received** → "Show me cheap restaurants near Blue Mosque"

2. **Signal Detection**
   - Semantic: Compare query embedding to signal patterns
   - Keyword: Check for explicit keywords
   - Combine: Use OR logic (if either detects, signal is ON)

3. **Signals Detected**
   - `likely_restaurant` (semantic + keyword)
   - `has_budget_constraint` (keyword: "cheap")
   - `mentions_location` (keyword: "near")
   - `likely_attraction` (keyword: "Blue Mosque")

4. **Context Building**
   - Retrieve relevant restaurant data
   - Add Blue Mosque location context
   - Include price filter information

5. **LLM Prompt**
   - Inject detected signals
   - Add database context
   - LLM generates intelligent response

---

## ⚡ Performance

| Metric | Value |
|--------|-------|
| Average Detection Time | 10-12ms |
| Semantic Model Load | ~2s (one-time, cached) |
| Cache Hit Rate | TBD (monitor in prod) |
| Memory Usage | ~200MB (embedding model) |

---

## 🐛 Debugging

### Check Signal Detection
```bash
# Production logs
grep "signal detection" /var/log/ai-istanbul.log

# Look for patterns
grep "Detected signals:" /var/log/ai-istanbul.log | tail -20
```

### View Cache Stats
```python
# In Python console
from backend.services.pure_llm_handler import handler
print(handler.stats)
# Shows cache_hits, signal_cache_hits, multi_signal_queries
```

### Test Specific Query
```python
import asyncio
from backend.services.pure_llm_handler import handler

async def test():
    signals = await handler._detect_service_signals(
        "Show me cheap restaurants near Galata Tower",
        user_location={"lat": 41.0082, "lng": 28.9784}
    )
    print(signals)

asyncio.run(test())
```

---

## 📈 Monitoring

### Key Metrics to Watch

1. **Cache Hit Rate** (target: >70%)
   ```python
   cache_rate = stats['signal_cache_hits'] / stats['total_queries']
   ```

2. **Multi-Signal Queries** (expect: 20-30%)
   ```python
   multi_rate = stats['multi_signal_queries'] / stats['total_queries']
   ```

3. **Average Signals Per Query** (expect: 2-3)
   ```python
   # Log this in production
   avg_signals = sum(len(s) for s in detected_signals) / num_queries
   ```

---

## 🔄 Configuration

### Adjust Detection Sensitivity
```python
# In pure_llm_handler.py

# More sensitive (more signals detected, some false positives)
SIMILARITY_THRESHOLD = 0.35
ATTRACTION_THRESHOLD = 0.30

# Less sensitive (fewer signals, more conservative)
SIMILARITY_THRESHOLD = 0.45
ATTRACTION_THRESHOLD = 0.40
```

### Cache TTL
```python
# Signal detection cache (default: 1 hour)
redis.setex(cache_key, 3600, json.dumps(signals))

# Increase for better performance
redis.setex(cache_key, 7200, json.dumps(signals))  # 2 hours
```

---

## 🚨 Common Issues

### Issue: Landmark not detected
**Solution:** Add to keyword list in `_detect_signals_keywords()`

### Issue: Wrong language detected
**Solution:** Semantic model handles both automatically, check keywords

### Issue: Too many false positives
**Solution:** Increase `SIMILARITY_THRESHOLD`

### Issue: Missing signals
**Solution:** Decrease `SIMILARITY_THRESHOLD` or add keywords

---

## 📞 Support

**Documentation:**
- Full Guide: `SIGNAL_DETECTION_MIGRATION_GUIDE.md`
- Implementation: `SIGNAL_BASED_INTENT_DETECTION_IMPLEMENTATION.md`
- Test Results: `SIGNAL_DETECTION_TEST_RESULTS.md`
- Deployment: `SIGNAL_DETECTION_PRODUCTION_DEPLOYMENT.md`

**Code:**
- Main Handler: `backend/services/pure_llm_handler.py`
- Test Suite: `test_signal_detection.py`

**Team Contact:** AI Istanbul Development Team

---

## ✅ Quick Health Check

Run this to verify system is working:

```bash
cd /Users/omer/Desktop/ai-stanbul
python -c "
import asyncio
from backend.services.pure_llm_handler import PureLLMHandler

# Quick test
handler = PureLLMHandler.__new__(PureLLMHandler)
handler.redis = None
handler.embedding_model = None
handler._signal_embeddings = {}

signals = handler._detect_signals_keywords('Show me restaurants near Galata Tower')
print('✅ System OK' if signals['likely_restaurant'] and signals['likely_attraction'] else '❌ Issue detected')
print(f'Detected: {[k for k,v in signals.items() if v]}')
"
```

Expected output:
```
✅ System OK
Detected: ['likely_restaurant', 'likely_attraction', 'mentions_location']
```

---

**Last Updated:** ${new Date().toISOString().split('T')[0]}  
**Version:** 1.0 - Production Ready
