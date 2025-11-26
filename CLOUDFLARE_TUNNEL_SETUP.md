# 🚀 CLOUDFLARE TUNNEL SETUP - FREE, NO SIGNUP NEEDED

**Run these commands in your RunPod SSH terminal:**

---

## Step 1: Install Cloudflared

```bash
cd /workspace
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared-linux-amd64
mv cloudflared-linux-amd64 cloudflared
```

---

## Step 2: Start Cloudflare Tunnel

```bash
./cloudflared tunnel --url http://localhost:8888
```

**You'll see output like this:**

```
2024-11-26T09:45:00Z INF Thank you for trying Cloudflare Tunnel. Doing so, without a Cloudflare account, is a quick way to experiment and try it out. However, be aware that these account-less Tunnels have no uptime guarantee. If you intend to use Tunnels in production you should use a pre-created named tunnel by following: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps
2024-11-26T09:45:00Z INF Requesting new quick Tunnel on trycloudflare.com...
2024-11-26T09:45:01Z INF +--------------------------------------------------------------------------------------------+
2024-11-26T09:45:01Z INF |  Your quick Tunnel has been created! Visit it at (it may take some time to be reachable):  |
2024-11-26T09:45:01Z INF |  https://abc-def-123.trycloudflare.com                                                     |
2024-11-26T09:45:01Z INF +--------------------------------------------------------------------------------------------+
```

**📋 COPY THE URL:** `https://abc-def-123.trycloudflare.com`

**⚠️ KEEP THIS TERMINAL OPEN!** The tunnel must stay running.

---

## Step 3: Test from Your Mac

**Open a NEW terminal on your Mac and replace with YOUR cloudflare URL:**

```bash
# Replace with YOUR actual Cloudflare URL!
export CF_URL="https://abc-def-123.trycloudflare.com"

# Test models
curl $CF_URL/v1/models | python3 -m json.tool

# Test completion
curl $CF_URL/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "prompt": "Tell me about Istanbul in one sentence:",
    "max_tokens": 100,
    "temperature": 0.7
  }' | python3 -m json.tool
```

**If you get JSON with AI-generated text, IT WORKS!** 🎉

---

## Step 4: Update Render Backend

1. Go to https://dashboard.render.com
2. Find your backend service
3. Click **Environment** tab
4. Add or update these variables:

```
RUNPOD_LLM_ENDPOINT=https://YOUR-CLOUDFLARE-URL-HERE
RUNPOD_LLM_MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
```

**Example (use YOUR URL):**
```
RUNPOD_LLM_ENDPOINT=https://abc-def-123.trycloudflare.com
RUNPOD_LLM_MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
```

5. Click **"Save Changes"**
6. Click **"Manual Deploy"** (top right)
7. Wait 2-3 minutes

---

## Step 5: Test Full Integration

**After Render finishes deploying:**

```bash
# Test backend health
curl https://api.aistanbul.net/api/v1/llm/health | python3 -m json.tool

# Test chat endpoint
curl -X POST https://api.aistanbul.net/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the Blue Mosque?",
    "language": "en"
  }' | python3 -m json.tool
```

**You should get a real AI response!** 🎉

---

## ✅ Success Checklist

- [ ] Installed cloudflared ✅
- [ ] Started tunnel ✅
- [ ] Copied Cloudflare URL ✅
- [ ] Tested from Mac ✅
- [ ] Updated Render environment variables ✅
- [ ] Redeployed backend ✅
- [ ] Tested backend health ✅
- [ ] Tested chat endpoint ✅
- [ ] Got real AI responses ✅

---

## 🔧 Alternative: Use screen to keep tunnel running

If you want to disconnect SSH but keep the tunnel running:

```bash
# Install screen (if not already)
apt-get update && apt-get install -y screen

# Start screen session
screen -S cloudflare

# Run cloudflared
/workspace/cloudflared tunnel --url http://localhost:8888

# Detach: Press Ctrl+A, then D
# Tunnel keeps running even if you disconnect SSH!

# Reattach later:
screen -r cloudflare
```

---

## 💡 Important Notes

1. **Free Cloudflare URLs** - No signup needed, works immediately
2. **URL changes** - Each time you restart cloudflared, you get a new URL
3. **Keep terminal open** - Or use screen/tmux to run in background
4. **Production** - For permanent URL, create a Cloudflare account (free)

---

## 🎯 Why Cloudflare is Better Than ngrok

✅ **No signup required** - Works immediately  
✅ **Completely free** - No trial limits  
✅ **Fast** - Cloudflare's global network  
✅ **Reliable** - Enterprise-grade infrastructure  

---

**Run Step 1-2 now and paste the Cloudflare URL here!** 🚀
