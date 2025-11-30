# LLM Response Speed Optimization Guide

## Current Performance

**Observed:** ~10 seconds per response  
**Target:** 3-5 seconds per response

---

## Speed Breakdown (Before Optimization)

| Component | Time | Details |
|-----------|------|---------|
| Network (Render → RunPod) | 1-2s | Proxy latency, HTTPS overhead |
| LLM Generation (250 tokens) | 7-8s | 4-bit AWQ quantization on GPU |
| Backend Processing | 0.5-1s | Context building, prompt construction |
| **Total** | **~10s** | |

---

## Optimization 1: Reduce Max Tokens ✅ (Applied)

**Changed:** `LLM_MAX_TOKENS` from 250 → 150

### Expected Impact:
- **Before:** 250 tokens × ~30ms/token = 7.5 seconds
- **After:** 150 tokens × ~30ms/token = 4.5 seconds
- **Savings:** ~3 seconds (40% faster generation)

### Why This Works:
- Most Istanbul queries need 100-150 tokens max
- The model was generating unnecessary extra tokens
- Shorter responses are often better UX anyway

### Update on Render:
```bash
LLM_MAX_TOKENS=150
```

---

## Optimization 2: Add Stop Sequences ✅ (Applied)

**Added to `runpod_llm_client.py`:**
```python
"stop": ["\n\nUser:", "\n\n---", "## User Question:", "## Your Response:"]
```

### Impact:
- Model stops as soon as it finishes the answer
- Prevents generating fake follow-up questions
- Saves 50-100 tokens per request
- **Additional savings:** 1-3 seconds

---

## Optimization 3: Simplified Prompt Format ✅ (Applied)

**Changed prompt ending from:**
```
## User Question:
{query}

## Your Response:
```

**To:**
```
---

User: {query}

Assistant:
```

### Impact:
- Cleaner format prevents template-style generation
- Model focuses on direct answer
- Reduces prompt complexity

---

## Expected Results After All Optimizations

### Before:
```json
{
  "response_time": "~10 seconds",
  "tokens_generated": "200-250",
  "issues": [
    "Generating fake Q&A examples",
    "Too verbose responses"
  ]
}
```

### After:
```json
{
  "response_time": "~4-6 seconds",
  "tokens_generated": "80-150",
  "improvements": [
    "Direct, concise answers",
    "No fake conversations",
    "40-50% faster"
  ]
}
```

---

## Additional Speed Optimizations (Optional)

### 4. Enable Response Streaming (Advanced)
**Status:** Already implemented in backend, not used in chat endpoint

**What it does:**
- Sends tokens to user as they're generated
- User sees response appearing in real-time
- **Perceived speed:** Instant (even if total time same)

**Implementation:** Already in `core.py` - need to use `process_query_stream()` instead of `process_query()`

### 5. Upgrade RunPod GPU (Costs Money)
Current: Likely RTX 3090 or similar  
Options:
- RTX 4090: ~50% faster ($0.40/hr vs $0.30/hr)
- A100 40GB: ~2x faster ($1.20/hr)

**For 8B model:** RTX 3090/4090 is perfectly fine

### 6. Use Smaller Model (Not Recommended)
- Llama 3.1 8B AWQ-INT4 is already optimized
- Smaller models (3B, 1B) would be faster but much worse quality
- Current model is the sweet spot

### 7. Increase Batch Size in vLLM (Minimal Impact)
Currently running with default batch size. For single-user traffic, won't help much.

---

## How to Update Render Environment

### Quick Steps:
1. Go to https://dashboard.render.com
2. Select your backend service
3. Go to **Environment** tab
4. Update:
   ```
   LLM_MAX_TOKENS=150
   ```
5. Save and wait for auto-redeploy (~2-3 minutes)

### Test After Update:
```bash
time curl -X POST https://api.aistanbul.net/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me about Istanbul briefly", "session_id": "test-123"}'
```

You should see response time drop from ~10s to ~5s

---

## Current Configuration Summary

| Setting | Value | Purpose |
|---------|-------|---------|
| Model | Meta-Llama-3.1-8B-Instruct-AWQ-INT4 | High quality, 4-bit quantized |
| Max Tokens | 150 (was 250) | Balance speed vs completeness |
| Temperature | 0.7 | Balanced creativity |
| Stop Sequences | 4 patterns | Prevent over-generation |
| Timeout | 60s | Safety limit |

---

## Monitoring Performance

### After deploying, check:
1. **Response times** - Should be 4-6 seconds
2. **Response quality** - Still complete and helpful?
3. **Token counts** - Check backend logs for actual tokens used

### If responses are too short:
- Increase to `LLM_MAX_TOKENS=200`

### If still too slow:
- Check RunPod GPU load: `nvidia-smi` on pod
- Verify no other processes using GPU
- Consider streaming for perceived speed

---

## Updated: November 30, 2025

**Status:**  
✅ Local `.env` updated (LLM_MAX_TOKENS=150)  
✅ Stop sequences added to client  
✅ Prompt format simplified  
❌ Render environment needs update (still at 250)  

**Next Step:** Update `LLM_MAX_TOKENS=150` on Render and test!
