# 🔧 Expose Port 8000 on RunPod

**Issue:** vLLM server is running on port 8000, but RunPod only shows ports 8888 and 19123 in HTTP Services.

**Pod:** `gbpd35labcq12f-64411272`

---

## ✅ Solution 1: Edit Pod to Add Port 8000 (Easiest)

1. Go to RunPod Console: https://www.runpod.io/console/pods
2. Find your pod: **gbpd35labcq12f-64411272**
3. Click the **⋮** (three dots) menu → **Edit Pod**
4. Look for **"Expose HTTP Ports"** or **"HTTP Service Ports"**
5. Add port: **8000**
6. Save/Apply changes
7. Wait ~30 seconds for RunPod to detect the service
8. Refresh the Connect tab - you should see **Port 8000** appear

**Expected URL format:**
```
https://gbpd35labcq12f-8000.proxy.runpod.net
```

---

## ✅ Solution 2: Use Direct TCP Port (Alternative)

Since you have direct TCP access via `194.68.245.16`, you can test if port 8000 is accessible:

**From your Mac terminal:**

```bash
# Try direct TCP access (if RunPod exposed it)
curl http://194.68.245.16:8000/v1/models
```

**If this doesn't work,** you'll need to use Solution 1 or 3.

---

## ✅ Solution 3: Use Port 8888 Instead (Quick Workaround)

Since port 8888 is already exposed (Jupyter), restart your vLLM server on port 8888:

**In your RunPod SSH:**

```bash
# Stop current server (Ctrl+C in the terminal where it's running)
# Or kill it
pkill -f vllm

# Restart on port 8888
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/llama-3.1-8b \
  --port 8888 \
  --host 0.0.0.0 \
  --trust-remote-code \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --dtype auto > /workspace/llm_server.log 2>&1 &

echo $! > /workspace/llm_server.pid
```

**Then test from your Mac:**

```bash
# The URL should be the one shown for port 8888
# But we need to find the correct format
curl https://gbpd35labcq12f-8888.proxy.runpod.net/v1/models
```

---

## ✅ Solution 4: Use ngrok or Similar (Advanced)

If RunPod doesn't expose custom ports easily, use a tunneling service:

**Install ngrok in RunPod:**

```bash
# Download ngrok
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar xvzf ngrok-v3-stable-linux-amd64.tgz

# Run ngrok to expose port 8000
./ngrok http 8000
```

This will give you a public URL like: `https://xxxx-xxx-xxx.ngrok-free.app`

---

## 🎯 Recommended Action

**Try Solution 1 first (Edit Pod):**

1. Go to RunPod Console
2. Edit your pod
3. Add port 8000 to exposed HTTP ports
4. Save and wait
5. Check Connect tab for the new URL

**If that doesn't work, use Solution 3 (restart on port 8888)** since that port is already exposed.

---

Let me know which solution you try and I'll help with the next steps! 🚀
