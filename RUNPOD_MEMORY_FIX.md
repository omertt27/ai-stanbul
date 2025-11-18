# 🔧 RunPod GPU Memory Fix Guide

## Problem
Your RTX A5000 (23.57 GB VRAM) only has 4.67 GB free. Something is using ~19 GB!

## Quick Fix Steps

### Step 1: Access RunPod Terminal
Open JupyterLab: https://fgkqzve33ssbea-19123.proxy.runpod.net/vagkyun4lkebh4kpvqm71rwz06mtn04w/
Then: **File → New → Terminal**

### Step 2: Check What's Using GPU Memory
```bash
nvidia-smi
```

Look for processes using GPU memory. Common culprits:
- Previous vLLM instances
- JupyterLab kernels
- Other Python processes

### Step 3: Kill GPU Processes
```bash
# Kill all vLLM processes
pkill -f vllm

# Kill all Python processes (be careful!)
pkill -f python

# Wait a moment
sleep 2

# Check again
nvidia-smi
```

### Step 4: Reset GPU (if needed)
```bash
nvidia-smi --gpu-reset
```

### Step 5: Start vLLM with Reduced Memory Settings

**Option A: Use the automated script** (recommended)
```bash
cd /workspace
wget https://raw.githubusercontent.com/YOUR-REPO/fix_runpod_vllm.sh
chmod +x fix_runpod_vllm.sh
bash fix_runpod_vllm.sh
```

**Option B: Manual command** (copy-paste this entire block)
```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --dtype half \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 \
  --port 8000
```

### Step 6: Verify It's Running
Wait 2-3 minutes for model loading. You should see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 7: Test from Your Local Machine
```bash
curl https://fgkqzve33ssbea-8000.proxy.runpod.net/v1/models
```

## Key Changes Made

| Setting | Before | After | Why |
|---------|--------|-------|-----|
| `--gpu-memory-utilization` | 0.90 (default) | 0.85 | Leaves more headroom |
| `--max-model-len` | 4096 | 2048 | Reduces KV cache memory |
| Process cleanup | None | Kill old processes | Frees GPU memory |

## Alternative: Even Lower Memory Usage

If you still have issues, try this **ultra-low memory mode**:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --dtype half \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.75 \
  --kv-cache-dtype fp8 \
  --host 0.0.0.0 \
  --port 8000
```

## Troubleshooting

### "Still out of memory"
1. Restart the entire RunPod instance
2. Use a smaller model like `Meta-Llama-3.1-8B` (no Instruct)
3. Try `--enforce-eager` to disable CUDA graphs

### "Model not found"
Make sure you're logged into Hugging Face:
```bash
huggingface-cli login
# Paste your token: hf_...
```

### "404 errors when testing"
The server takes 2-3 minutes to start. Be patient!

## Success Indicators

✅ GPU memory usage: ~15-18 GB (not 19+ GB)
✅ vLLM logs show "Application startup complete"
✅ `curl` returns model information
✅ No errors in logs

---

**Need help?** Share the output of `nvidia-smi` and the vLLM startup logs.
