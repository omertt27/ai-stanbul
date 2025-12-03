# 🚀 RUN THESE COMMANDS ON RUNPOD TERMINAL

## Option 1: Use RunPod Web Terminal

Go to your RunPod dashboard and click **"Open Web Terminal"**

---

## Option 2: SSH into RunPod

```bash
ssh -p 22003 -i ~/.ssh/id_ed25519 root@194.68.245.153
```

---

## Then Run These Commands:

### Step 1: Go to workspace
```bash
cd /workspace
```

### Step 2: Install dependencies (with --break-system-packages for Ubuntu 24.04)
```bash
pip install --break-system-packages fastapi uvicorn[standard] transformers torch accelerate bitsandbytes pydantic requests
```

### Step 3: Kill any existing server
```bash
pkill -f "llm_server.py" 2>/dev/null || true
```

### Step 4: Create logs directory
```bash
mkdir -p /workspace/logs
```

### Step 5: Start server with nohup
```bash
nohup python /workspace/llm_server.py > /workspace/logs/llm_server.log 2>&1 &
```

### Step 6: Save PID and disown
```bash
echo $! > /workspace/llm_server.pid
disown
```

### Step 7: Wait for model to load (30-60 seconds)
```bash
echo "Waiting for model to load..."
sleep 60
```

### Step 8: Test server
```bash
curl http://localhost:8000/health | python3 -m json.tool
```

### Step 9: Test completion
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Istanbul is", "max_tokens": 30}' \
  | python3 -m json.tool
```

### Step 10: Check GPU
```bash
nvidia-smi
```

---

## 📋 All Commands in One Block (Copy & Paste):

```bash
cd /workspace && \
pip install --break-system-packages fastapi uvicorn[standard] transformers torch accelerate bitsandbytes pydantic requests && \
pkill -f "llm_server.py" 2>/dev/null || true && \
sleep 2 && \
mkdir -p /workspace/logs && \
nohup python /workspace/llm_server.py > /workspace/logs/llm_server.log 2>&1 & \
echo $! > /workspace/llm_server.pid && \
disown && \
echo "✅ Server started! Waiting 60 seconds for model to load..." && \
sleep 60 && \
curl http://localhost:8000/health | python3 -m json.tool
```

---

## 🔍 After Server Starts:

### View logs:
```bash
tail -f /workspace/logs/llm_server.log
```

### Check if running:
```bash
ps aux | grep llm_server.py
```

### Check GPU usage:
```bash
nvidia-smi
```

### Stop server:
```bash
kill $(cat /workspace/llm_server.pid)
```

---

## ✅ What to Expect:

1. Dependencies install (~30 seconds)
2. Server starts in background
3. Model loads into GPU (~30-60 seconds)
4. Health endpoint returns JSON
5. Server keeps running even after you disconnect!

---

**Just copy the "All Commands in One Block" section and paste it into RunPod terminal!** 🚀
