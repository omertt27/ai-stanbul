# 🚨 RUNPOD WORKSPACE LOST - QUICK RECOVERY GUIDE

**Problem:** RunPod workspace was lost after restart  
**Solution:** Reinstall vLLM and download model (15 minutes)

---

## 🎯 QUICK SETUP (Run in RunPod Terminal)

### Step 1: Install vLLM (2 minutes)
```bash
pip install vllm
```

### Step 2: Download Model (5-10 minutes)
```bash
cd /workspace
huggingface-cli download hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --local-dir /workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
```

### Step 3: Start vLLM Server (1 minute)
```bash
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.85 \
  --max-model-len 2048 \
  --port 8888 \
  --host 0.0.0.0 \
  > /workspace/vllm.log 2>&1 &
```

### Step 4: Monitor Startup (2-3 minutes)
```bash
# Watch the logs
tail -f /workspace/vllm.log

# Wait for: "Application startup complete"
# Press Ctrl+C to exit tail
```

### Step 5: Test vLLM
```bash
curl -X POST http://localhost:8888/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    "prompt": "Hello",
    "max_tokens": 10
  }'

# Should return: JSON with completion text
```

---

## 📋 FULL STEP-BY-STEP COMMANDS

Copy and paste these into your RunPod terminal:

```bash
# 1. Install vLLM
pip install vllm

# 2. Create workspace directory
cd /workspace

# 3. Download model (this takes 5-10 minutes)
huggingface-cli download hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --local-dir /workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4

# 4. Start vLLM in background
nohup python -m vllm.entrypoints.openai.api_server \
  --model /workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.85 \
  --max-model-len 2048 \
  --port 8888 \
  --host 0.0.0.0 \
  > /workspace/vllm.log 2>&1 &

# 5. Wait 2-3 minutes, then check logs
tail -f /workspace/vllm.log

# 6. Test (in new terminal or after Ctrl+C)
curl http://localhost:8888/health
```

---

## 🔧 AFTER SETUP COMPLETE

### Get Your RunPod Endpoint

1. Find your Pod ID in RunPod dashboard
2. Your endpoint will be:
   ```
   https://YOUR-POD-ID-8888.proxy.runpod.net/v1
   ```

### Update Backend Environment Variable

1. Go to https://dashboard.render.com
2. Find backend service
3. Update `LLM_API_URL`:
   ```
   https://YOUR-POD-ID-8888.proxy.runpod.net/v1
   ```
4. Save (triggers redeploy)

---

## ⏰ TIMELINE

- **Install vLLM:** 2 minutes
- **Download model:** 5-10 minutes
- **Start vLLM:** 1 minute
- **Wait for startup:** 2-3 minutes
- **Update backend:** 2 minutes
- **Test:** 1 minute

**Total:** ~15-20 minutes

---

## 🆘 TROUBLESHOOTING

### If model download is slow
```bash
# Download may take 10-15 minutes on slow connections
# Model size: ~4.5 GB
# Be patient and let it complete
```

### If vLLM fails to start
```bash
# Check logs
cat /workspace/vllm.log

# Check GPU
nvidia-smi

# Restart vLLM
pkill -f vllm
# Then run start command again
```

### If port 8888 is busy
```bash
# Check what's using it
lsof -i :8888

# Kill it
pkill -f "port 8888"
```

---

## 💡 PRO TIP: Keep vLLM Running

Use `screen` to keep vLLM running even if you disconnect:

```bash
# Install screen (if needed)
apt-get update && apt-get install -y screen

# Start screen session
screen -S vllm

# Run vLLM (without nohup)
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.85 \
  --max-model-len 2048 \
  --port 8888 \
  --host 0.0.0.0

# Detach from screen: Ctrl+A, then D

# Reattach later:
screen -r vllm
```

---

## ✅ VERIFICATION

After setup, verify everything works:

```bash
# 1. Check vLLM process
ps aux | grep vllm

# 2. Test health endpoint
curl http://localhost:8888/health

# 3. Test completion
curl -X POST http://localhost:8888/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    "prompt": "What is Istanbul?",
    "max_tokens": 50
  }'

# 4. Test external endpoint (from your computer)
curl https://YOUR-POD-ID-8888.proxy.runpod.net/health
```

---

## 🔗 NEXT STEPS

After vLLM is running:

1. ✅ Note your Pod ID
2. ✅ Update `LLM_API_URL` on Render
3. ✅ Wait for backend redeploy (2-3 min)
4. ✅ Test chat API
5. ✅ Rotate database password (security)

---

**Status:** 🔴 vLLM needs to be reinstalled  
**Time Required:** 15-20 minutes  
**Priority:** HIGH - Chat won't work until vLLM is running
