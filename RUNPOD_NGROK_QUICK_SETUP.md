# 🚀 RunPod ngrok Setup - Fixed Commands

vLLM is running perfectly on port 8000! Now let's expose it with ngrok.

---

## Step 1: Install ngrok on RunPod

Run these commands in your RunPod terminal:

```bash
# Download ngrok (correct URL)
cd /tmp
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz

# Extract
tar xvzf ngrok-v3-stable-linux-amd64.tgz

# Move to system path
mv ngrok /usr/local/bin/

# Make executable
chmod +x /usr/local/bin/ngrok

# Verify installation
ngrok version
```

---

## Step 2: Get ngrok Auth Token (Optional but Recommended)

1. Go to: https://dashboard.ngrok.com/signup
2. Sign up (free account)
3. Copy your auth token from: https://dashboard.ngrok.com/get-started/your-authtoken

Then on RunPod:

```bash
ngrok config add-authtoken YOUR_TOKEN_HERE
```

**OR skip this step and use ngrok without auth (limited to 1 connection)**

---

## Step 3: Start ngrok Tunnel

```bash
# Start ngrok in background
nohup ngrok http 8000 > /workspace/ngrok.log 2>&1 &

# Wait a moment
sleep 5

# Get the public URL
curl -s http://localhost:4040/api/tunnels | grep -o 'https://[^"]*\.ngrok-free\.app'
```

**Copy the URL** that appears (e.g., `https://abc-123-def.ngrok-free.app`)

---

## Step 4: Test Your Public Endpoint

**From your Mac terminal**, test the ngrok URL:

```bash
# Replace with YOUR actual ngrok URL
export NGROK_URL="https://abc-123-def.ngrok-free.app"

# Test health
curl $NGROK_URL/health

# Test models
curl $NGROK_URL/v1/models

# Test chat completion
curl -X POST $NGROK_URL/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    "messages": [
      {"role": "user", "content": "What are the top places to visit in Istanbul?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

**If you see JSON responses with Istanbul info, it works!** ✅

---

## Step 5: Update Your Backend (Render)

1. Go to: https://dashboard.render.com
2. Find your backend service
3. Click **Environment** tab
4. Add or update this variable:

```
LLM_API_URL=https://YOUR-NGROK-URL.ngrok-free.app/v1
```

**Important:** Add `/v1` at the end!

Example:
```
LLM_API_URL=https://abc-123-def.ngrok-free.app/v1
```

5. Click **Save Changes**
6. Render will automatically redeploy (wait 2-3 minutes)

---

## Step 6: Test Your Chat App!

After Render finishes deploying, test your frontend:

1. Open your Istanbul AI chat app
2. Send a message: "What are the best restaurants in Istanbul?"
3. You should get a **real AI-generated response**! 🎉

---

## 🔍 Monitor ngrok

**View ngrok dashboard:**
```bash
# On RunPod, forward the dashboard port
curl http://localhost:4040

# Or check logs
tail -f /workspace/ngrok.log
```

**Check if ngrok is still running:**
```bash
ps aux | grep ngrok
```

**Restart ngrok if needed:**
```bash
pkill ngrok
nohup ngrok http 8000 > /workspace/ngrok.log 2>&1 &
```

---

## ⚠️ Important Notes

### ngrok Free Tier:
- ✅ Unlimited HTTP requests
- ✅ Works great for development/testing
- ⚠️ URL changes if you restart ngrok
- ⚠️ Session timeout after 2 hours (reconnects automatically)

### ngrok Paid ($8/mo):
- ✅ Custom domain (doesn't change)
- ✅ No session timeout
- ✅ Better for production

---

## 🎯 Quick Summary

Your setup will be:
```
Frontend (Browser)
    ↓
Render Backend (api.aistanbul.net)
    ↓
ngrok URL (https://abc-123.ngrok-free.app/v1)
    ↓
RunPod vLLM (Meta Llama 3.1 8B - 4-bit)
```

---

**Start with Step 1 on RunPod!** 🚀
