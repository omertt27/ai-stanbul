# 🔧 FIX RENDER ENVIRONMENT VARIABLES - DO THIS NOW!

## ❌ Problem:
Backend is using OLD URL: `https://9llrfrwmscmmth-8888.proxy.runpod.net/v1`  
It should use NEW URL: `https://miller-researchers-girls-college.trycloudflare.com`

---

## ✅ Solution: Update Render with CORRECT Variable Names

### Step 1: Go to Render Dashboard

```
https://dashboard.render.com
```

### Step 2: Find Your Backend Service

Look for your backend service and click on it.

### Step 3: Click "Environment" Tab

In the left sidebar, click **"Environment"**

### Step 4: Update/Add These Variables

**IMPORTANT: Use these EXACT variable names!**

#### Variable 1:
**Key:** `LLM_API_URL`  
**Value:** `https://miller-researchers-girls-college.trycloudflare.com/v1`

⚠️ **NOTE:** Add `/v1` at the end!

#### Variable 2:
**Key:** `LLM_MODEL_NAME`  
**Value:** `meta-llama/Meta-Llama-3.1-8B-Instruct`

### Step 5: Remove or Update Old Variables (if they exist)

If you see these old variables, either DELETE them or UPDATE them:
- `RUNPOD_LLM_ENDPOINT` → DELETE or change to correct URL
- `RUNPOD_LLM_MODEL_NAME` → DELETE or change to correct name
- `LLM_API_URL` → UPDATE to new Cloudflare URL

### Step 6: Save and Deploy

1. Click **"Save Changes"**
2. Click **"Manual Deploy"** (top right)
3. Wait 2-3 minutes

---

## 🧪 Test After Deployment

Run these commands on your Mac:

```bash
# Test 1: Health check
curl https://api.aistanbul.net/api/v1/llm/health | python3 -m json.tool

# Should show:
# "status": "healthy"
# "llm_available": true
```

```bash
# Test 2: Chat
curl -X POST https://api.aistanbul.net/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tell me about Istanbul",
    "language": "en"
  }' | python3 -m json.tool

# Should show real AI response!
```

---

## 📋 Correct Environment Variables Summary

Copy these EXACT values to Render:

```
LLM_API_URL=https://miller-researchers-girls-college.trycloudflare.com/v1
LLM_MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
```

⚠️ **IMPORTANT:**
- Variable name is `LLM_API_URL` (not `RUNPOD_LLM_ENDPOINT`)
- URL must end with `/v1`
- Model name must be exact

---

## 🎯 Quick Checklist:

- [ ] Opened Render dashboard
- [ ] Found backend service
- [ ] Clicked "Environment" tab
- [ ] Added/Updated `LLM_API_URL` with Cloudflare URL + `/v1`
- [ ] Added/Updated `LLM_MODEL_NAME` with model name
- [ ] Removed or updated old variables
- [ ] Clicked "Save Changes"
- [ ] Clicked "Manual Deploy"
- [ ] Waited for deployment (2-3 min)
- [ ] Tested health endpoint
- [ ] Tested chat endpoint
- [ ] Got real AI response! 🎉

---

**Go to Render NOW and fix these variables!** 🚀
