# ✅ CONNECTED TO RUNPOD - SETUP NGROK NOW

**You're in RunPod! Now run these commands in order:**

---

## Step 1: Verify Server is Running

```bash
curl http://localhost:8888/health
```

**Should return:** `OK`

```bash
curl http://localhost:8888/v1/models
```

**Should show:** `meta-llama/Meta-Llama-3.1-8B-Instruct`

---

## Step 2: Download and Setup ngrok

```bash
cd /workspace
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar xvzf ngrok-v3-stable-linux-amd64.tgz
chmod +x ngrok
```

---

## Step 3: Start ngrok

```bash
./ngrok http 8888
```

**You'll see something like this:**

```
ngrok                                                                                        

Session Status                online                                                         
Account                       (Plan: Free)                                                   
Version                       3.x.x                                                          
Region                        United States (us)                                             
Latency                       -                                                              
Web Interface                 http://127.0.0.1:4040                                          
Forwarding                    https://a1b2-c3d4-e5f6.ngrok-free.app -> http://localhost:8888

Connections                   ttl     opn     rt1     rt5     p50     p90                    
                              0       0       0.00    0.00    0.00    0.00
```

**📋 COPY THE URL:** `https://a1b2-c3d4-e5f6.ngrok-free.app`

**⚠️ KEEP THIS TERMINAL OPEN!** ngrok must stay running.

---

## Step 4: Test from Your Mac

**Open a NEW terminal on your Mac and replace with YOUR ngrok URL:**

```bash
# Replace with YOUR actual ngrok URL!
export NGROK_URL="https://a1b2-c3d4-e5f6.ngrok-free.app"

# Test models
curl $NGROK_URL/v1/models | python3 -m json.tool

# Test completion
curl $NGROK_URL/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "prompt": "Tell me about Istanbul:",
    "max_tokens": 100,
    "temperature": 0.7
  }' | python3 -m json.tool
```

**If you get JSON responses with text about Istanbul, IT WORKS!** 🎉

---

## Step 5: Update Render Backend

1. Go to https://dashboard.render.com
2. Find your backend service
3. Click **Environment** tab
4. Add or update these variables:

```
RUNPOD_LLM_ENDPOINT=https://YOUR-NGROK-URL-HERE
RUNPOD_LLM_MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
```

**Example (use YOUR URL):**
```
RUNPOD_LLM_ENDPOINT=https://a1b2-c3d4-e5f6.ngrok-free.app
RUNPOD_LLM_MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
```

5. Click **"Save Changes"**
6. Click **"Manual Deploy"** (top right)
7. Wait 2-3 minutes

---

## Step 6: Test Full Integration

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

**You should get a real AI response about the Blue Mosque!** 🎉

---

## 🎯 Quick Summary

1. ✅ Server is running on RunPod (port 8888)
2. 🔄 Setup ngrok to expose it publicly
3. 📋 Copy the ngrok URL
4. 🧪 Test from your Mac
5. 🚀 Update Render with ngrok URL
6. ✅ Test full integration

**Start with Step 1 to verify your server, then do Step 2-3!**
