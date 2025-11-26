# 🔴 CLOUDFLARE TUNNEL IS DOWN!

## Problem:
The Cloudflare tunnel at `https://miller-researchers-girls-college.trycloudflare.com` is no longer accessible.

This means:
1. ❌ You closed the RunPod SSH terminal with cloudflared running, OR
2. ❌ The cloudflared process stopped, OR  
3. ❌ You restarted cloudflared and got a NEW URL

---

## ✅ Solution: Restart Cloudflare Tunnel on RunPod

### Step 1: SSH into RunPod

```bash
ssh rjexqr4adxw135-644111dc@ssh.runpod.io -i ~/.ssh/id_ed25519
```

### Step 2: Check if vLLM server is still running

```bash
curl http://localhost:8888/v1/models
```

**If this works**, server is running! If not, restart it:

```bash
# Check if it's running
ps aux | grep vllm

# If not running, restart:
nohup python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8888 \
  --host 0.0.0.0 \
  --trust-remote-code \
  --max-model-len 4096 > /workspace/llm_server.log 2>&1 &
```

### Step 3: Restart Cloudflare Tunnel

```bash
cd /workspace
./cloudflared tunnel --url http://localhost:8888
```

You'll see a **NEW URL** like:
```
https://something-different-words.trycloudflare.com
```

**📋 COPY THIS NEW URL!**

### Step 4: Update Render Backend

1. Go to: https://dashboard.render.com
2. Click your backend service
3. Click **"Environment"**
4. Update `LLM_API_URL` with the NEW Cloudflare URL + `/v1`:

```
LLM_API_URL=https://NEW-URL-HERE.trycloudflare.com/v1
```

5. Click **"Save Changes"**
6. Click **"Manual Deploy"**
7. Wait 2-3 minutes

### Step 5: Test Again

```bash
curl https://api.aistanbul.net/api/v1/llm/health | python3 -m json.tool
```

---

## 🎯 BETTER SOLUTION: Use Screen to Keep Tunnel Running

To prevent this from happening again:

### In RunPod SSH:

```bash
# Install screen
apt-get update && apt-get install -y screen

# Start screen session
screen -S cloudflare

# Run cloudflared
cd /workspace
./cloudflared tunnel --url http://localhost:8888

# Press Ctrl+A, then D to detach
# Now you can close SSH and tunnel stays running!

# To reconnect later:
screen -r cloudflare
```

---

## 📋 Quick Commands Summary:

```bash
# 1. SSH to RunPod
ssh rjexqr4adxw135-644111dc@ssh.runpod.io -i ~/.ssh/id_ed25519

# 2. Restart cloudflared
cd /workspace && ./cloudflared tunnel --url http://localhost:8888

# 3. Copy the new URL

# 4. Update Render:
#    LLM_API_URL=https://NEW-URL.trycloudflare.com/v1

# 5. Deploy on Render

# 6. Test
curl https://api.aistanbul.net/api/v1/llm/health | python3 -m json.tool
```

---

**SSH into RunPod NOW and restart the tunnel!** 🚀
