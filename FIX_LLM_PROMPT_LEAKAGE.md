# URGENT: Fix LLM Response Issue + Database 🚨

## Current Problem

The LLM is returning **system prompt instructions** instead of a proper response:

```
"response": " Use the hybrid approach and follow the critical rules for accuracy..."
```

This is **prompt leakage** - the LLM is parroting back its instructions instead of following them.

## Fixes Applied

### 1. ✅ Database Connection Fix
Fixed `/Users/omer/Desktop/ai-stanbul/backend/database.py` to convert `postgres://` → `postgresql://`

### 2. ✅ Prompt Engineering Fix  
Simplified `/Users/omer/Desktop/ai-stanbul/backend/services/llm/prompts.py`:
- Removed confusing example responses from system prompt
- Made instructions clearer and more direct
- LLM should now generate real responses, not echo instructions

## Deploy Both Fixes NOW

```bash
cd /Users/omer/Desktop/ai-stanbul

# Check what files changed
git status

# Add both fixes
git add backend/database.py backend/services/llm/prompts.py

# Commit
git commit -m "fix: database connection and prompt leakage issue"

# Deploy
git push origin main
```

## What Should Happen After Deploy

1. **Render rebuilds** (2-3 minutes)
2. **Database connects to PostgreSQL** (not localhost)
3. **LLM generates real responses** (not prompt echo)
4. **Test response should look like:**

```json
{
  "response": "I recommend these budget-friendly Turkish restaurants in Sultanahmet:\n\n1. **Sultanahmet Köftecisi** - $\n   Traditional Turkish meatballs, local favorite since 1920\n   📍 Divan Yolu Caddesi No: 12\n   ⭐ 4.5/5\n\n2. **Tarihi Sultanahmet Köftecisi** - $\n   Authentic Turkish cuisine, great köfte\n   📍 Alemdar Mah, Divan Yolu Cad\n   ⭐ 4.3/5",
  "session_id": "...",
  "intent": "restaurant",
  "confidence": 0.95
}
```

## After Deploy - Test Immediately

```bash
# Wait 2-3 minutes for Render to deploy, then:

curl -X POST https://ai-stanbul.onrender.com/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "recommend a cheap Turkish restaurant in Sultanahmet",
    "language": "en"
  }'
```

## Expected Results ✅

- ✅ Response is a **real recommendation**, not prompt instructions
- ✅ Prices show as **$** (not TL or price ranges)
- ✅ Includes 2-3 specific restaurants with details
- ✅ Database connection works (no localhost error in logs)
- ✅ All 12 services loaded successfully

## If Still Getting Prompt Leakage

The issue might also be:

### A. Model Configuration
Check LLM generation parameters:
- Temperature might be too low (causing repetition)
- Max tokens might be too low
- Model needs better stop sequences

### B. Check vLLM Server
On RunPod:
```bash
# Check vLLM logs
curl http://localhost:8888/v1/models

# Try direct generation test
curl http://localhost:8888/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "Recommend a restaurant in Istanbul:",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### C. Alternative: Use Chat Format
Llama 3.1 Instruct models work better with chat format. We might need to switch from `/v1/completions` to `/v1/chat/completions`.

## Deploy Command (Quick Copy-Paste)

```bash
cd /Users/omer/Desktop/ai-stanbul && \
git add backend/database.py backend/services/llm/prompts.py && \
git commit -m "fix: database connection and prompt leakage" && \
git push origin main && \
echo "✅ Deployed! Check Render logs in 2-3 minutes, then test chat endpoint"
```

---

**🎯 DO THIS NOW:**
1. Run the deploy command above
2. Wait 2-3 minutes
3. Test the chat endpoint
4. Share the response here

If the response still shows prompt leakage after deploy, we'll need to check the vLLM configuration and possibly switch to chat format.
