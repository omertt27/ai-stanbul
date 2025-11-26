# 🚀 RUNPOD NGROK SETUP - EASIEST SOLUTION

**Your server is running! Now let's make it accessible with ngrok.**

---

## Step 1: SSH into RunPod (Use Correct Credentials)

```bash
ssh rjexqr4adxw135-644111dc@ssh.runpod.io -i ~/.ssh/id_ed25519
```

**Or if that doesn't work:**

```bash
ssh rjexqr4adxw135-644111dc@ssh.runpod.io
```

Enter password when prompted.

---

## Step 2: Setup ngrok Inside RunPod

**Once connected to RunPod, run these commands:**

```bash
# Go to workspace
cd /workspace

# Download ngrok
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz

# Extract it
tar xvzf ngrok-v3-stable-linux-amd64.tgz

# Start ngrok tunnel
./ngrok http 8888
```

---

## Step 3: Copy Your Public URL

You'll see output like this:

```
ngrok                                                                                        

Session Status                online                                                         
Account                       (Plan: Free)                                                   
Version                       3.x.x                                                          
Region                        United States (us)                                             
Latency                       -                                                              
Web Interface                 http://127.0.0.1:4040                                          
Forwarding                    https://1a2b-3c4d-5e6f.ngrok-free.app -> http://localhost:8888

Connections                   ttl     opn     rt1     rt5     p50     p90                    
                              0       0       0.00    0.00    0.00    0.00
```

**Copy the `https://xxxxx.ngrok-free.app` URL!**

**⚠️ KEEP THIS TERMINAL OPEN!** If you close it, the tunnel stops.

---

## Step 4: Test From Your Mac

**Open a NEW terminal on your Mac:**

```bash
# Replace with YOUR ngrok URL
export NGROK_URL="https://1a2b-3c4d-5e6f.ngrok-free.app"

# Test models endpoint
curl $NGROK_URL/v1/models | python3 -m json.tool

# Test completion
curl $NGROK_URL/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "prompt": "Istanbul is famous for",
    "max_tokens": 50,
    "temperature": 0.7
  }' | python3 -m json.tool
```

**If you see JSON responses, IT WORKS!** ✅

---

## Step 5: Update Render Backend

1. Go to https://dashboard.render.com
2. Find your backend service (probably "ai-stanbul-backend" or similar)
3. Click **Environment** tab
4. Update/add these variables:

```
RUNPOD_LLM_ENDPOINT=https://YOUR-NGROK-URL-HERE
RUNPOD_LLM_MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
```

**Example:**
```
RUNPOD_LLM_ENDPOINT=https://1a2b-3c4d-5e6f.ngrok-free.app
RUNPOD_LLM_MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
```

5. Click **"Save Changes"**
6. Click **"Manual Deploy"**
7. Wait 2-3 minutes for deployment

---

## Step 6: Test Full Integration

**After Render deploys:**

```bash
# Test backend health
curl https://api.aistanbul.net/api/v1/llm/health | python3 -m json.tool

# Test chat
curl -X POST https://api.aistanbul.net/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tell me about the Blue Mosque",
    "language": "en"
  }' | python3 -m json.tool
```

**You should get a REAL AI response about the Blue Mosque!** 🎉

---

## ✅ Success Checklist

- [ ] SSH into RunPod with correct credentials ✅
- [ ] Downloaded ngrok ✅
- [ ] Started ngrok tunnel ✅
- [ ] Copied ngrok URL ✅
- [ ] Tested from Mac ✅
- [ ] Updated Render environment variables ✅
- [ ] Redeployed backend ✅
- [ ] Tested backend health ✅
- [ ] Tested chat endpoint ✅
- [ ] Got real AI responses ✅

---

## 🔧 Troubleshooting

### ngrok download fails

```bash
# Try alternative download
curl -O https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
```

### ngrok shows "tunnel not found"

Make sure port 8888 server is running:
```bash
curl http://localhost:8888/health
```

If it fails, restart the server:
```bash
kill $(cat /workspace/llm_server.pid)
# Then run Step 7 from main guide again
```

### ngrok closes when I disconnect SSH

Use `screen` or `tmux`:
```bash
# Start screen session
screen -S ngrok

# Run ngrok
./ngrok http 8888

# Detach: Press Ctrl+A, then D
# Server keeps running even if you disconnect!

# Reattach later:
screen -r ngrok
```

---

## 💡 Important Notes

1. **Free ngrok URLs change** - Each time you restart ngrok, you get a new URL
2. **Update Render** - You need to update Render env vars if URL changes
3. **Keep terminal open** - ngrok needs to stay running
4. **Alternative** - Paid ngrok ($8/month) gives you a fixed URL

---

## 🎯 What You'll Have After This

✅ RunPod running Llama 3.1 8B  
✅ ngrok making it publicly accessible  
✅ Render backend connected to LLM  
✅ Frontend → Backend → LLM working end-to-end  
✅ Real AI responses in your app!

---

**Start with Step 1 and work through each step!**
