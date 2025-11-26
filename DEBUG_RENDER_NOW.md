# 🔍 DEBUG RENDER BACKEND - CHECK THIS NOW

## Step 1: Check Render Logs

1. Go to: https://dashboard.render.com
2. Click on your backend service
3. Click **"Logs"** tab (left sidebar)
4. Look for these messages in the logs:

### ✅ Good Signs (what you SHOULD see):
```
🚀 Starting AI Istanbul Backend
⚡ Initializing Pure LLM Handler with resilience features...
🔍 LLM_API_URL from env: https://miller-researchers-girls-college.trycloudflare.com/v1
✅ RunPod LLM Client initialized: https://miller-researchers-girls-college.trycloudflare.com/v1
✅ Pure LLM Core initialized with circuit breakers and resilience patterns
✅ Backend startup complete
```

### ❌ Bad Signs (what might be wrong):
```
⚠️ Pure LLM mode disabled
```
If you see this, it means `PURE_LLM_MODE` is not set to `true`!

```
⚠️ RunPod LLM Client created but disabled (no URL)
```
If you see this, it means `LLM_API_URL` is not being read correctly!

---

## Step 2: Verify Environment Variables

1. Stay in Render dashboard
2. Click **"Environment"** tab (left sidebar)
3. **Check these EXACT variable names and values:**

```
LLM_API_URL=https://miller-researchers-girls-college.trycloudflare.com/v1
```
⚠️ Must end with `/v1`!

```
LLM_MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
```

```
PURE_LLM_MODE=true
```
⚠️ Must be lowercase `true`!

---

## Step 3: Force Redeploy

Even if variables are set, sometimes Render needs a forced redeploy:

1. Click **"Manual Deploy"** button (top right)
2. Select **"Clear build cache & deploy"**
3. Wait 3-5 minutes
4. Watch the logs carefully for the messages above

---

## Step 4: Test Again

After deployment completes:

```bash
# Wait 30 seconds after "Deploy live" message
sleep 30

# Then test
curl https://api.aistanbul.net/api/v1/llm/health | python3 -m json.tool
```

Should show:
```json
{
    "status": "healthy",
    "llm_available": true,
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct"
}
```

---

## 🆘 If Still Not Working:

### Check 1: Is Cloudflare tunnel still running?

In your RunPod SSH terminal, you should still see:
```
INF Registered tunnel connection
```

If not, restart it:
```bash
cd /workspace
./cloudflared tunnel --url http://localhost:8888
```

### Check 2: Test Cloudflare URL directly

```bash
curl https://miller-researchers-girls-college.trycloudflare.com/v1/models
```

Should return JSON with model info.

### Check 3: Check for typos

Common mistakes:
- ❌ `PURE_LLM_MODE=True` (capital T) → ✅ Should be `true`
- ❌ `LLM_API_URL=...com` (missing /v1) → ✅ Should end with `/v1`
- ❌ Extra spaces or newlines in variables
- ❌ Wrong variable names (RUNPOD_LLM_ENDPOINT instead of LLM_API_URL)

---

## 📋 Quick Checklist:

- [ ] Opened Render dashboard
- [ ] Clicked "Logs" tab
- [ ] Searched for "🚀 Starting AI Istanbul"
- [ ] Checked if you see "Pure LLM mode disabled" (BAD)
- [ ] Checked if you see "RunPod LLM Client initialized" (GOOD)
- [ ] Verified all 3 environment variables are set correctly
- [ ] PURE_LLM_MODE is lowercase `true`
- [ ] LLM_API_URL ends with `/v1`
- [ ] Did "Clear build cache & deploy"
- [ ] Waited for deployment to complete
- [ ] Tested health endpoint

---

**Check the Render logs NOW and tell me what you see!**

Specifically look for:
1. Do you see "⚠️ Pure LLM mode disabled"?
2. Do you see "✅ RunPod LLM Client initialized"?
3. What messages do you see during startup?
