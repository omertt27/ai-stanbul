# 🎯 VLLM PORT MISMATCH DETECTED - CRITICAL FIX NEEDED

**Date:** November 30, 2025  
**Status:** 🔴 URGENT - Port configuration mismatch

---

## 🔍 PROBLEM IDENTIFIED

Your vLLM server is running on **port 8888**, but the backend is likely configured to use **port 8000**.

### Current State

**✅ vLLM Status (WORKING):**
```bash
# Running on port 8888
python -m vllm.entrypoints.openai.api_server --port 8888

# Test succeeds on port 8888
curl http://localhost:8888/v1/completions
# Returns: Valid completion ✅
```

**❌ Backend Configuration (WRONG PORT):**
```bash
# Likely configured with port 8000
LLM_API_URL=https://fcn3h0wk2vf5sk-8000.proxy.runpod.net/v1
```

**⚠️ Result:**
- Backend tries to connect to port 8000
- vLLM is actually running on port 8888
- Requests fail → Fallback errors

---

## 🔧 IMMEDIATE FIX (5 minutes)

### Option A: Update Backend Environment Variable (RECOMMENDED)

**1. Go to Render Dashboard**
```
https://dashboard.render.com
```

**2. Update Backend Service Environment Variables**
```
Find: LLM_API_URL
Change FROM: https://fcn3h0wk2vf5sk-8000.proxy.runpod.net/v1
Change TO:   https://fcn3h0wk2vf5sk-8888.proxy.runpod.net/v1
                                         ↑↑↑↑
                                      Change 8000 to 8888
```

**3. Save and Wait for Redeploy** (2-3 minutes)

**4. Test**
```bash
curl -X POST "https://ai-stanbul.onrender.com/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello Istanbul!"}'
```

### Option B: Restart vLLM on Port 8000

**On RunPod terminal:**
```bash
# 1. Stop current vLLM process
pkill -f vllm

# 2. Start vLLM on port 8000 instead of 8888
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.85 \
  --max-model-len 2048 \
  --port 8000 \
  --host 0.0.0.0 \
  > /workspace/vllm.log 2>&1 &

# 3. Verify
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "/workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4", "prompt": "Test", "max_tokens": 10}'
```

---

## 🎯 RECOMMENDED SOLUTION

**👉 OPTION A** - Update backend environment variable

**Why?**
- Faster (no need to restart vLLM)
- Safer (vLLM is already working)
- Less disruptive

**Timeline:**
- 2 min: Update environment variable on Render
- 2-3 min: Wait for backend redeploy
- 1 min: Test and verify
- **Total: 5 minutes**

---

## 📋 VERIFICATION CHECKLIST

After fixing:

```bash
# 1. Check vLLM is accessible
curl https://fcn3h0wk2vf5sk-8888.proxy.runpod.net/v1/models

# 2. Test backend chat API
curl -X POST "https://ai-stanbul.onrender.com/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the best restaurants in Kadıköy?"}'

# 3. Check response
# Should get: Real AI response (not fallback error)
```

**Expected Results:**
- ✅ Backend connects to vLLM successfully
- ✅ Chat returns real AI-generated responses
- ✅ No more "I apologize, but I'm having trouble..." errors

---

## 🔗 RUNPOD URLS TO UPDATE

**Your RunPod Pod ID:** `fcn3h0wk2vf5sk`

**Correct URLs (port 8888):**
```
HTTP (internal): http://localhost:8888/v1
HTTPS (external): https://fcn3h0wk2vf5sk-8888.proxy.runpod.net/v1
Health check: https://fcn3h0wk2vf5sk-8888.proxy.runpod.net/health
Models: https://fcn3h0wk2vf5sk-8888.proxy.runpod.net/v1/models
```

**Environment Variable to Set on Render:**
```bash
LLM_API_URL=https://fcn3h0wk2vf5sk-8888.proxy.runpod.net/v1
```

---

## 🧪 TEST COMMANDS

### Test vLLM Directly (from RunPod)
```bash
curl -X POST http://localhost:8888/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    "prompt": "Tell me about Istanbul",
    "max_tokens": 50
  }'
```

### Test vLLM via Proxy (from anywhere)
```bash
curl -X POST https://fcn3h0wk2vf5sk-8888.proxy.runpod.net/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    "prompt": "Tell me about Istanbul",
    "max_tokens": 50
  }'
```

### Test Backend Chat API
```bash
curl -X POST https://ai-stanbul.onrender.com/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the weather in Istanbul?"}'
```

---

## 📊 TROUBLESHOOTING

### If still getting fallback errors after port fix:

**1. Check Render environment variable**
```
Go to Render Dashboard → Backend Service → Environment
Verify: LLM_API_URL ends with :8888
```

**2. Check vLLM is accessible externally**
```bash
curl https://fcn3h0wk2vf5sk-8888.proxy.runpod.net/health
# Should return: {"status": "ok"} or similar
```

**3. Check backend logs on Render**
```
Look for:
✅ "Connected to LLM service"
❌ "Failed to connect to LLM"
❌ "Connection refused"
```

**4. Clear cache (if needed)**
```bash
# Restart backend on Render to clear cache
Manual Deploy → Deploy latest commit
```

---

## 🎯 NEXT STEPS

1. **[2 min] NOW:** Update LLM_API_URL on Render (port 8000 → 8888)
2. **[3 min] WAIT:** For backend to redeploy
3. **[2 min] TEST:** Chat API with unique query
4. **[5 min] VERIFY:** Frontend chat works end-to-end

**Total Time:** ~12 minutes

---

## 🔐 BONUS: After This Fix

Don't forget to also:
- [ ] Rotate PostgreSQL password (security issue)
- [ ] Update DATABASE_URL with new credentials
- [ ] Clean Git history to remove exposed credentials

See: `SECURITY_BREACH_QUICK_FIX.md`

---

**Status:** 🎯 ROOT CAUSE IDENTIFIED - Port mismatch  
**Solution:** Update LLM_API_URL to use port 8888  
**Priority:** 🔴 HIGH - Fix immediately  
**Estimated Fix Time:** 5 minutes
