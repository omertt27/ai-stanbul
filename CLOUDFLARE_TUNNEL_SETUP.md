# 🚀 CLOUDFLARE TUNNEL SETUP - FREE, NO SIGNUP NEEDED

**Run these commands in your RunPod SSH terminal (or Web Terminal)**

---

## 🎯 Quick Start (2 Options)

### Option A: Quick Tunnel (Temporary - for testing)
Fast setup, but URL changes each restart

### Option B: Persistent Tunnel with nohup
Runs in background, survives disconnect (recommended for production)

---

## Option A: Quick Tunnel (Testing)

### Step 1: Install Cloudflared

```bash
cd /workspace
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared-linux-amd64
mv cloudflared-linux-amd64 cloudflared
```

### Step 2: Start Tunnel (blocks terminal)

```bash
./cloudflared tunnel --url http://localhost:8000
```

### Step 2: Start Tunnel (blocks terminal)

```bash
./cloudflared tunnel --url http://localhost:8000
```

**You'll see output like this:**

```
2024-12-03T10:20:00Z INF |  Your quick Tunnel has been created! Visit it at:  |
2024-12-03T10:20:00Z INF |  https://abc-def-123.trycloudflare.com              |
2024-12-03T10:20:00Z INF +--------------------------------------------------------------------------------------------+
```

**📋 COPY THE URL:** `https://abc-def-123.trycloudflare.com`

**⚠️ KEEP THIS TERMINAL OPEN!** The tunnel must stay running.

---

## Option B: Persistent Tunnel with nohup (RECOMMENDED)

### Step 1: Install Cloudflared (if not done)

```bash
cd /workspace
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared-linux-amd64
mv cloudflared-linux-amd64 /usr/local/bin/cloudflared
```

### Step 2: Start Tunnel in Background

```bash
cd /workspace
mkdir -p logs

# Start tunnel with nohup
nohup cloudflared tunnel --url http://localhost:8000 > /workspace/logs/cloudflare-tunnel.log 2>&1 &

# Save PID
echo $! > /workspace/cloudflare-tunnel.pid

# Disown to keep running
disown

echo "✅ Cloudflare Tunnel started in background!"
```

### Step 3: Get Your Tunnel URL

```bash
# Wait a few seconds for tunnel to start
sleep 5

# View the log to get your URL
cat /workspace/logs/cloudflare-tunnel.log | grep "trycloudflare.com"
```

**Look for a line like:**
```
https://abc-def-123.trycloudflare.com
```

**📋 COPY THIS URL!**

### Step 4: Test from RunPod

```bash
# Replace with your actual URL
curl https://your-actual-url.trycloudflare.com/health | python3 -m json.tool
```

---

## 🎯 All-in-One Command (Copy & Paste)

```bash
cd /workspace && \
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && \
chmod +x cloudflared-linux-amd64 && \
mv cloudflared-linux-amd64 /usr/local/bin/cloudflared && \
mkdir -p logs && \
nohup cloudflared tunnel --url http://localhost:8000 > /workspace/logs/cloudflare-tunnel.log 2>&1 & \
echo $! > /workspace/cloudflare-tunnel.pid && \
disown && \
echo "✅ Tunnel starting... Wait 5 seconds..." && \
sleep 5 && \
echo "📋 Your Cloudflare URL:" && \
grep -o "https://.*trycloudflare.com" /workspace/logs/cloudflare-tunnel.log | head -1
```

This command:
1. ✅ Downloads cloudflared
2. ✅ Installs it system-wide
3. ✅ Starts tunnel with nohup
4. ✅ Saves PID for management
5. ✅ Shows your tunnel URL

---

## Step 3: Test from Your Local Machine

**From your Mac terminal:**

```bash
# Replace with YOUR actual Cloudflare URL!
export CF_URL="https://your-actual-url.trycloudflare.com"

# Test health
curl $CF_URL/health | python3 -m json.tool

# Test completion
curl -X POST $CF_URL/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Istanbul is",
    "max_tokens": 50,
    "temperature": 0.7
  }' | python3 -m json.tool
```

**If you get JSON with AI-generated text, IT WORKS!** 🎉

---

## 🛠️ Tunnel Management

### View Tunnel URL:
```bash
grep -o "https://.*trycloudflare.com" /workspace/logs/cloudflare-tunnel.log | head -1
```

### Check if Running:
```bash
ps aux | grep cloudflared
```

### View Logs:
```bash
tail -f /workspace/logs/cloudflare-tunnel.log
```

### Stop Tunnel:
```bash
kill $(cat /workspace/cloudflare-tunnel.pid)
```

### Restart Tunnel:
```bash
cd /workspace && \
nohup cloudflared tunnel --url http://localhost:8000 > /workspace/logs/cloudflare-tunnel.log 2>&1 & \
echo $! > /workspace/cloudflare-tunnel.pid && \
disown && \
sleep 5 && \
grep -o "https://.*trycloudflare.com" /workspace/logs/cloudflare-tunnel.log | head -1
```

---

## Step 4: Update Your Backend

Add the Cloudflare URL to your backend configuration:

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
