# 🚀 START vLLM on Port 8000 - EXECUTE NOW

## ✅ Confirmed Configuration
- **Port**: 8000
- **Pod ID**: 4r1su4zfuok0s7
- **Public URL**: `https://4r1su4zfuok0s7-8000.proxy.runpod.net`

## 📋 Commands to Run in RunPod Terminal

You're already in RunPod SSH. Copy and paste these commands:

### 1️⃣ Check Available Models

```bash
ls -lh /workspace/
```

### 2️⃣ Start vLLM on Port 8000

```bash
cd /workspace

nohup python -m vllm.entrypoints.openai.api_server \
  --model /workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.85 \
  --max-model-len 2048 \
  --port 8000 \
  --host 0.0.0.0 \
  > /workspace/vllm.log 2>&1 &
```

**Note**: If your model has a different name, adjust the `--model` path.

Expected output:
```
[1] 12345
nohup: ignoring input and appending output to 'vllm.log'
```

### 3️⃣ Watch Startup (2-3 Minutes)

```bash
tail -f /workspace/vllm.log
```

Wait for:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Then press **Ctrl+C** to exit (vLLM keeps running).

### 4️⃣ Test Locally on RunPod

```bash
# Check if running
ps aux | grep vllm

# Test health endpoint
curl http://localhost:8000/health

# Test completion
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    "prompt": "Hello!",
    "max_tokens": 20
  }'
```

### 5️⃣ Exit SSH

```bash
exit
```

vLLM will keep running in the background.

## 🧪 Test from Your Mac

Now on your Mac terminal, test the public endpoint:

```bash
# Test health
curl https://4r1su4zfuok0s7-8000.proxy.runpod.net/health

# Test completion
curl -X POST https://4r1su4zfuok0s7-8000.proxy.runpod.net/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    "prompt": "Say hello to Istanbul!",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

## ⚙️ Update .env File

I'll update your `.env` file with the correct port 8000 URL:

```bash
AI_ISTANBUL_LLM_MODE=runpod
PURE_LLM_MODE=true
LLM_API_URL=https://4r1su4zfuok0s7-8000.proxy.runpod.net
```

## 🎯 Your Endpoints

- **Local (on pod)**: `http://localhost:8000`
- **Public (from Mac)**: `https://4r1su4zfuok0s7-8000.proxy.runpod.net`

## ✅ Quick Checklist

- [ ] Run `cd /workspace` in RunPod
- [ ] Run the `nohup python -m vllm...` command
- [ ] Wait for "Application startup complete"
- [ ] Test with `curl http://localhost:8000/health`
- [ ] Exit SSH with `exit`
- [ ] Test from Mac: `curl https://4r1su4zfuok0s7-8000.proxy.runpod.net/health`
- [ ] Confirm it returns `{"status":"healthy",...}`

Once you confirm it works, I'll update the `.env` file! 🚀
