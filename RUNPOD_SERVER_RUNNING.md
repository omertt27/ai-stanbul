# ✅ vLLM Server is Running!

**Status:** Server successfully started on port 8000

**Pod:** `gbpd35labcq12f-64411272`

---

## 🎯 Next Steps (Run These Commands)

### ✅ Step 1: Test Server Health (Inside RunPod SSH)

```bash
curl http://localhost:8000/health
```

**Expected output:** `{"status":"ok"}` or similar

---

### ✅ Step 2: List Available Models

```bash
curl http://localhost:8000/v1/models | python3 -m json.tool
```

**Expected output:** JSON with model details (meta-llama/Meta-Llama-3.1-8B-Instruct or Qwen)

---

### ✅ Step 3: Test Chat Completion (Optional)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }' | python3 -m json.tool
```

---

### ✅ Step 4: Find Your Public RunPod URL

**Option A: From RunPod Console (Easiest)**
1. Go to RunPod Console: https://www.runpod.io/console/pods
2. Click on your pod: `gbpd35labcq12f-64411272`
3. Click the **"Connect"** tab
4. Look for **"HTTP Service [Port 8000]"** or **"TCP Port Mappings"**
5. Copy the URL (format: `https://gbpd35labcq12f-64411272-8000.proxy.runpod.net`)

**Option B: Via SSH (Check logs)**
```bash
# Sometimes the URL is in the pod info
cat /etc/runpod_config.json 2>/dev/null || echo "Config not found"
```

---

### ✅ Step 5: Test Public URL (From Your Mac)

**Open a NEW terminal on your Mac** (not SSH) and run:

```bash
# Replace with your actual URL from Step 4
curl https://gbpd35labcq12f-64411272-8000.proxy.runpod.net/health

curl https://gbpd35labcq12f-64411272-8000.proxy.runpod.net/v1/models
```

**Expected:** Same responses as Step 1 & 2, but from public internet

---

## 🔗 Backend Integration

Once you have the public URL, update your backend:

### Update Environment Variables

```bash
# In your backend .env or environment
LLM_API_URL=https://gbpd35labcq12f-64411272-8000.proxy.runpod.net
PURE_LLM_MODE=true
```

### Test Backend Health

```bash
# From your Mac
curl https://your-backend-url.com/health
curl https://your-backend-url.com/chat -X POST -H "Content-Type: application/json" -d '{"message":"test"}'
```

---

## 🔍 Monitoring Commands (Inside RunPod SSH)

**Check if server is still running:**
```bash
ps aux | grep vllm
```

**View live logs:**
```bash
tail -f /workspace/llm_server.log
```

**Check GPU usage:**
```bash
nvidia-smi
```

**Server process ID:**
```bash
cat /workspace/llm_server.pid
```

---

## 🛑 Stop/Restart Server

**Stop server:**
```bash
kill $(cat /workspace/llm_server.pid)
```

**Restart server (Llama 3.1 8B):**
```bash
nohup python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8000 \
  --host 0.0.0.0 \
  --trust-remote-code \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 > /workspace/llm_server.log 2>&1 &

echo $! > /workspace/llm_server.pid
```

---

## 📋 Success Checklist

- ✅ Server started successfully
- ⬜ Health endpoint working (localhost)
- ⬜ Models endpoint working (localhost)
- ⬜ Public URL obtained from RunPod Console
- ⬜ Public URL tested from Mac
- ⬜ Backend environment variables updated
- ⬜ Backend health check passing
- ⬜ Backend chat endpoint returning real LLM responses

---

## 🎯 Current Task

**Run Step 1 now to verify the server is responding:**

```bash
curl http://localhost:8000/health
```

Then proceed through the remaining steps! 🚀
