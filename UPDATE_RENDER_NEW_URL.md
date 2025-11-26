# ✅ NEW CLOUDFLARE TUNNEL URL - UPDATE RENDER NOW

## Your New Tunnel URL:
```
https://announced-novel-excellence-carolina.trycloudflare.com
```

---

## 🚀 Update Render Backend (DO THIS NOW):

### Step 1: Go to Render Dashboard
```
https://dashboard.render.com
```

### Step 2: Find Your Backend Service
Click on your backend service (ai-stanbul-backend or similar)

### Step 3: Update Environment Variable
1. Click **"Environment"** in left sidebar
2. Find `LLM_API_URL`
3. Change it to:
```
LLM_API_URL=https://announced-novel-excellence-carolina.trycloudflare.com/v1
```

⚠️ **IMPORTANT:** Must end with `/v1`!

4. Click **"Save Changes"**

### Step 4: Deploy
1. Click **"Manual Deploy"** (top right)
2. Wait 2-3 minutes for deployment

---

## 🧪 Test After Deployment (wait 30 seconds after "Deploy live"):

```bash
# Test 1: Health
curl https://api.aistanbul.net/api/v1/llm/health | python3 -m json.tool
```

**Should show:**
```json
{
    "status": "healthy",
    "llm_available": true,
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct"
}
```

```bash
# Test 2: Chat
curl -X POST https://api.aistanbul.net/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tell me about Hagia Sophia in one sentence",
    "language": "en"
  }' | python3 -m json.tool
```

**Should show:** Real AI response about Hagia Sophia!

---

## 📋 Quick Summary:

**Old URL (dead):**
```
https://miller-researchers-girls-college.trycloudflare.com
```

**New URL (active):**
```
https://announced-novel-excellence-carolina.trycloudflare.com
```

**Render Variable:**
```
LLM_API_URL=https://announced-novel-excellence-carolina.trycloudflare.com/v1
```

---

## 🎯 Next Steps:

1. ✅ Got new tunnel URL
2. ⏳ Update Render (DO NOW)
3. ⏳ Deploy backend
4. ⏳ Test health endpoint
5. ⏳ Test chat endpoint
6. 🎉 Done!

---

**Go to Render NOW and update the environment variable!** 🚀
