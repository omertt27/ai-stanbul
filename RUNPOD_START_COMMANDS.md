# 🚀 START vLLM ON RUNPOD - COPY/PASTE THIS

## ISSUE DETECTED: vLLM is NOT installed!
❌ ModuleNotFoundError: No module named 'vllm'

---

## STEP 1: INSTALL vLLM (Copy this first)

```bash
pip install vllm==0.5.4 && echo "✅ vLLM installed!"
```

**Wait for installation to complete** (takes 2-3 minutes)

---

## STEP 2: START vLLM SERVER (Copy this after Step 1)

```bash
export HF_HOME=/workspace/.cache && \
export TRANSFORMERS_CACHE=/workspace/.cache && \
cd /workspace && \
python -m vllm.entrypoints.openai.api_server \
  --model solidrust/Llama-3.1-8B-Instruct-AWQ \
  --quantization awq \
  --dtype half \
  --max-model-len 2048 \
  --host 0.0.0.0 \
  --port 8888 \
  --gpu-memory-utilization 0.85 \
  > vllm.log 2>&1 & \
echo "✅ vLLM starting... PID: $!" && \
sleep 10 && \
tail -50 vllm.log
```

---

## What This Does:
1. Changes to `/workspace` directory
2. Starts vLLM server with Llama 3.1 8B
3. Uses full precision (float16) - NO quantization needed (model is already optimized)
4. Listens on port 8888
5. Logs to `vllm.log`
6. Shows first 30 lines of logs after 5 seconds

---

## Expected Output (after 2-5 minutes):

```
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8888 (Press CTRL+C to quit)
```

---

## After Startup, Run These Tests:

### Test 1: Health Check (Local)
```bash
curl http://localhost:8888/health
```
**Expected:** `{"status":"ok"}` or similar

### Test 2: Model Info (Local)
```bash
curl http://localhost:8888/v1/models
```
**Expected:** JSON with model info

### Test 3: Chat Completion (Local)
```bash
curl -X POST http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/llama-3.1-8b",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 50
  }'
```
**Expected:** JSON response with LLM answer

---

## Once Tests Pass, Run This on Your Mac:

```bash
# Test public endpoint
curl https://gbpd35labcq12f-8888.proxy.runpod.net/health

# Test chat completion
curl -X POST https://gbpd35labcq12f-8888.proxy.runpod.net/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/llama-3.1-8b",
    "messages": [{"role": "user", "content": "What is Istanbul?"}],
    "max_tokens": 100
  }'
```

---

## If You See Errors:

### "Address already in use"
```bash
# Kill any existing process on port 8888
lsof -ti:8888 | xargs kill -9
# Then restart vLLM
```

### "CUDA out of memory"
```bash
# Use less memory
cd /workspace && python -m vllm.entrypoints.openai.api_server --model /workspace/llama-3.1-8b --dtype float16 --max-model-len 2048 --host 0.0.0.0 --port 8888 --gpu-memory-utilization 0.8 > vllm.log 2>&1 &
```

### "Module not found"
```bash
# Install vLLM
pip install vllm
```

---

## Monitor Progress:

```bash
# Watch logs continuously
tail -f /workspace/vllm.log

# Check if vLLM is running
ps aux | grep vllm

# Check listening ports (install if needed)
apt-get update && apt-get install -y net-tools && netstat -tuln | grep 8888
```

---

## ⚡ QUICK START (All-in-One Command):

**Just copy this ONE line:**

```bash
cd /workspace && python -m vllm.entrypoints.openai.api_server --model /workspace/llama-3.1-8b --dtype float16 --max-model-len 4096 --host 0.0.0.0 --port 8888 --gpu-memory-utilization 0.9 > vllm.log 2>&1 & echo "vLLM PID: $!" && echo "Waiting 10s..." && sleep 10 && echo "=== LOGS ===" && tail -50 vllm.log && echo "=== TEST ===" && sleep 30 && curl http://localhost:8888/health
```

This will:
- Start vLLM
- Wait 10 seconds
- Show logs
- Wait 30 more seconds
- Test health endpoint

---

## 🎯 What to Tell Me After:

1. ✅ "vLLM started, logs show: Uvicorn running on..."
2. ✅ "Local test passed: curl localhost:8888/health returned..."
3. ✅ "Public test passed: curl https://gbpd35labcq12f-8888.proxy.runpod.net/health returned..."

Then I'll immediately update the backend and complete the integration! 🚀
