# ⚠️ URGENT: Start vLLM on Your RunPod

## Current Status
✅ Backend is configured and ready
✅ Backend is pointing to the correct RunPod URL
❌ **vLLM server is NOT running on your RunPod**

All test requests to your RunPod returned errors:
- Port 8888: `502 Bad Gateway` (port exposed but nothing listening)
- Port 8000: No response
- Port 19123: `404 Not Found`

## Quick Fix: Start vLLM Now

### Step 1: SSH into Your RunPod

Open your terminal and connect:

```bash
# The URL you provided suggests port 19123 for SSH
# In RunPod console, look for "Connect via SSH" and copy the command
# It should look something like:
ssh root@ssh.runpod.io -p XXXXX -i ~/.ssh/id_ed25519
```

Or use the web terminal:
1. Go to https://www.runpod.io/console/pods
2. Click on your pod
3. Click "Connect" → "Start Web Terminal"

### Step 2: Check Current Status

```bash
# Check if vLLM is running
ps aux | grep vllm

# Check listening ports
netstat -tuln | grep LISTEN

# Check if model exists
ls -lah /workspace/llama-3.1-8b
```

### Step 3: Start vLLM Server

**Option A: Start on Port 8888 (Recommended)**

```bash
cd /workspace

# Start vLLM with AWQ 4-bit quantization
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/llama-3.1-8b \
  --quantization awq \
  --dtype float16 \
  --max-model-len 4096 \
  --host 0.0.0.0 \
  --port 8888 \
  --gpu-memory-utilization 0.9 \
  > vllm.log 2>&1 &

echo "vLLM starting... Check logs with: tail -f /workspace/vllm.log"
```

**Option B: Start on Port 8000 (Alternative)**

```bash
cd /workspace

python -m vllm.entrypoints.openai.api_server \
  --model /workspace/llama-3.1-8b \
  --quantization awq \
  --dtype float16 \
  --max-model-len 4096 \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  > vllm.log 2>&1 &

echo "vLLM starting... Check logs with: tail -f /workspace/vllm.log"
```

### Step 4: Monitor Startup

```bash
# Watch the logs (this will take 2-5 minutes)
tail -f /workspace/vllm.log

# Look for this line:
# "Uvicorn running on http://0.0.0.0:8888 (Press CTRL+C to quit)"
```

**Expected Output:**
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8888 (Press CTRL+C to quit)
```

Press `Ctrl+C` to stop watching logs (vLLM will keep running).

### Step 5: Test vLLM Locally on RunPod

```bash
# Test health endpoint
curl http://localhost:8888/health

# Test chat endpoint
curl -X POST http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/llama-3.1-8b",
    "messages": [{"role": "user", "content": "Hello, what is Istanbul?"}],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

You should see a JSON response with the LLM's answer!

### Step 6: Find Your Public Endpoint

Once vLLM is running, your public endpoint will be:

```
# If using port 8888:
https://gbpd35labcq12f-8888.proxy.runpod.net/v1

# If using port 8000:
https://gbpd35labcq12f-8000.proxy.runpod.net/v1
```

### Step 7: Test Public Endpoint from Your Mac

From your local machine:

```bash
# Test port 8888
curl -X POST https://gbpd35labcq12f-8888.proxy.runpod.net/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/llama-3.1-8b",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50
  }'
```

### Step 8: Update Backend (I'll do this once you confirm the port)

Once you confirm which port works, let me know and I'll:
1. Update the `.env` file with the correct URL
2. Restart the backend
3. Test the full integration

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'vllm'"

Install vLLM:
```bash
pip install vllm
```

### Error: "Model not found"

Download the model:
```bash
cd /workspace
huggingface-cli login  # Use your HF token
huggingface-cli download TheBloke/Llama-3.1-8B-AWQ --local-dir llama-3.1-8b
```

### Error: "CUDA out of memory"

Reduce memory usage:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/llama-3.1-8b \
  --quantization awq \
  --dtype float16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.8 \
  --host 0.0.0.0 \
  --port 8888
```

### Error: "Port already in use"

Kill existing process:
```bash
# Find the process
lsof -ti:8888

# Kill it
kill -9 $(lsof -ti:8888)

# Or kill all vllm processes
pkill -f vllm
```

## What You Need to Tell Me

Once you've started vLLM, please provide:

1. **Which port did you use?** (8888 or 8000)
2. **Confirmation that local test works:**
   ```bash
   curl http://localhost:8888/health
   ```
3. **Confirmation that public test works:**
   ```bash
   curl https://gbpd35labcq12f-8888.proxy.runpod.net/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "/workspace/llama-3.1-8b", "messages": [{"role": "user", "content": "test"}], "max_tokens": 20}'
   ```

Then I'll complete the integration! 🚀

## Quick Command Summary

```bash
# 1. SSH into RunPod
# 2. Start vLLM
cd /workspace
python -m vllm.entrypoints.openai.api_server --model /workspace/llama-3.1-8b --quantization awq --dtype float16 --max-model-len 4096 --host 0.0.0.0 --port 8888 --gpu-memory-utilization 0.9 > vllm.log 2>&1 &

# 3. Watch logs
tail -f vllm.log

# 4. Test locally
curl http://localhost:8888/health

# 5. Test publicly
curl https://gbpd35labcq12f-8888.proxy.runpod.net/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "/workspace/llama-3.1-8b", "messages": [{"role": "user", "content": "test"}], "max_tokens": 20}'

# 6. Tell me which port works! 🎯
```
