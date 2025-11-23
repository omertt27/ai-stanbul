# 🚀 Start Llama 3.1 8B Instruct (4-bit) on RunPod

**Model**: meta-llama/Meta-Llama-3.1-8B-Instruct  
**Quantization**: 4-bit (for faster inference on GPU)  
**Server**: vLLM (OpenAI-compatible API)

---

## ⚠️ IMPORTANT: Requirements!

### 🔑 1. HuggingFace Token (REQUIRED)
**Llama 3.1 8B requires a HuggingFace token to download.**

1. Go to https://huggingface.co/settings/tokens
2. Create a new token (or copy existing one)
3. Accept Llama 3.1 terms: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct

### 💾 2. Disk Space (REQUIRED)
**You need 25GB+ free disk space**

Check your RunPod space first:
```bash
df -h
```

**If you don't have 25GB free:** See [RUNPOD_DISK_SPACE_FIX.md](./RUNPOD_DISK_SPACE_FIX.md) for solutions:
- Use smaller model (Qwen 2.5 7B needs only 15GB)
- Change cache location to larger disk
- Clean up old files
- Upgrade RunPod instance

---

## ⚡ Quick Start (Copy-Paste in RunPod SSH)

### ⚠️ Check disk space first!
```bash
df -h
# Need 25GB+ free. If not, see RUNPOD_DISK_SPACE_FIX.md
```

### Option 1: One-Line Command (if 25GB+ free)
```bash
# Set your HuggingFace token (REQUIRED!)
export HF_TOKEN="hf_YOUR_TOKEN_HERE"

# Use container disk (usually has more space)
export HF_HOME=/root/.cache/huggingface

# Install vLLM and start server
pip install vllm && python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8888 \
  --host 0.0.0.0 \
  --trust-remote-code \
  --max-model-len 4096 &
```

**Note:** Model download takes 5-10 minutes first time (16GB download). Wait for "Application startup complete" message.

### Option 1B: Smaller Model (if only 15GB free)
Use Qwen 2.5 7B instead - equally good, half the size:
```bash
export HF_TOKEN="hf_YOUR_TOKEN_HERE"

pip install vllm && python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 8888 \
  --host 0.0.0.0 \
  --trust-remote-code \
  --max-model-len 4096 &
```

Test after model loads:
```bash
# Wait for download/startup
sleep 300  # 5 minutes - adjust if needed

# Test endpoints
curl http://localhost:8888/health
curl http://localhost:8888/v1/models
```

---

### Option 2: Step-by-Step with Pre-download

#### 1️⃣ Set HuggingFace Token (REQUIRED)
```bash
export HF_TOKEN="hf_YOUR_TOKEN_HERE"
```

#### 2️⃣ Install requirements
```bash
pip install vllm huggingface_hub
```

#### 3️⃣ Download model first (track progress)
```bash
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct --local-dir ./llama-3.1-8b
```

#### 4️⃣ Start server from local path
```bash
python -m vllm.entrypoints.openai.api_server \
  --model ./llama-3.1-8b \
  --port 8888 \
  --host 0.0.0.0 \
  --trust-remote-code \
  --max-model-len 4096 &
```

#### 5️⃣ Wait for model to load and test
```bash
# Wait for "Application startup complete" message
sleep 60

# Health check
curl http://localhost:8888/health

# List models
curl http://localhost:8888/v1/models | python3 -m json.tool

# Test generation
curl http://localhost:8888/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "prompt": "Hello, how are you?",
    "max_tokens": 50
  }'
```

---

## 📋 What Each Flag Does

| Flag | Purpose |
|------|---------|
| `--model` | HuggingFace model ID or local path |
| `--port 8888` | Server port (matches your config) |
| `--host 0.0.0.0` | Accept connections from anywhere |
| `--trust-remote-code` | Allow model to run custom code (required for Llama) |
| `--max-model-len 4096` | Max context length (tokens) |

**Note:** `--quantization awq` removed - vLLM handles quantization automatically for this model.

---

## ✅ Expected Output

### When server starts (first 30 seconds):
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Loading model meta-llama/Meta-Llama-3.1-8B-Instruct...
INFO:     Model loaded successfully
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8888
```

### Health check response:
```bash
$ curl http://localhost:8888/health
"OK"
```

### Models endpoint response:
```json
{
  "object": "list",
  "data": [
    {
      "id": "Meta-Llama-3.1-8B-Instruct",
      "object": "model",
      "created": 1700000000,
      "owned_by": "vllm"
    }
  ]
}
```

---

## 🔧 Troubleshooting

### Issue: "OSError: [Errno 122] Disk quota exceeded"
**MOST COMMON ISSUE!** Not enough disk space.

**Solution:** See [RUNPOD_DISK_SPACE_FIX.md](./RUNPOD_DISK_SPACE_FIX.md) for complete guide.

**Quick fix:**
```bash
# Check space
df -h

# Clear old cache
rm -rf ~/.cache/huggingface/hub/*

# Use container disk instead
export HF_HOME=/root/.cache/huggingface
export HF_TOKEN="hf_YOUR_TOKEN_HERE"

# Try again
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8888 --host 0.0.0.0 --trust-remote-code &
```

**If still not enough space, use smaller model:**
```bash
# Qwen 2.5 7B only needs 15GB
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 8888 --host 0.0.0.0 --trust-remote-code &
```

---

### Issue: "401 Unauthorized" or "Repo is gated"
**Solution:** Set HuggingFace token and accept terms:
```bash
# 1. Get token from: https://huggingface.co/settings/tokens
# 2. Accept terms: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
# 3. Set token
export HF_TOKEN="hf_YOUR_TOKEN_HERE"

# 4. Then start server
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8888 \
  --host 0.0.0.0 \
  --trust-remote-code
```

### Issue: "No module named 'vllm'"
```bash
pip install vllm
```

### Issue: "CUDA out of memory"
Try with smaller max length:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8888 \
  --host 0.0.0.0 \
  --trust-remote-code \
  --max-model-len 2048 &  # Reduced from 4096
```

### Issue: Model download is slow
First time download can take 5-10 minutes (8GB+ model). Monitor progress:
```bash
# Check download progress
watch -n 5 'du -sh ~/.cache/huggingface/hub'

# Or use pre-download method (Option 2 above)
```

### Issue: Server not responding after starting
Check if still downloading or loading:
```bash
# Check server process
ps aux | grep vllm

# Check logs
tail -f /tmp/vllm-*.log

# Check network
netstat -tuln | grep 8888
```

Run in foreground to see all output:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8888 \
  --host 0.0.0.0 \
  --trust-remote-code
```

---

## 🎯 After Server Starts Successfully

### 1. Keep server running persistently
The `&` runs server in background, but it will stop if SSH disconnects.

**For persistent server (survives SSH disconnect):**
```bash
export HF_TOKEN="hf_YOUR_TOKEN_HERE"

nohup python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8888 \
  --host 0.0.0.0 \
  --trust-remote-code \
  --max-model-len 4096 > /workspace/llm_server.log 2>&1 &

echo $! > /workspace/llm_server.pid
```

Now you can safely disconnect SSH!

### 2. Test from your local machine
```bash
# Test RunPod proxy URL
curl https://ytc61lal7ag5sy-19123.proxy.runpod.net/2feph6uogs25wg1sc0i37280ah5ajfmm/v1/models
```

### 3. Verify Render environment variable
Make sure `LLM_API_URL` in Render Dashboard has the hyphen and no newline:
```
https://ytc61lal7ag5sy-19123.proxy.runpod.net/2feph6uogs25wg1sc0i37280ah5ajfmm/v1
```

### 4. Redeploy backend in Render
Go to Render Dashboard → **Manual Deploy** → Wait 2-3 minutes

### 5. Verify full integration
```bash
# Test backend LLM health
curl https://api.aistanbul.net/api/v1/llm/health | python3 -m json.tool

# Test chat endpoint with real LLM
curl -X POST https://api.aistanbul.net/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Tell me about Hagia Sophia in Istanbul","language":"en"}' \
  | python3 -m json.tool
```

Look for real LLM response (not fallback "I understand...")!

---

## 📊 System Requirements

**For Llama 3.1 8B 4-bit:**
- GPU: 8GB+ VRAM (should work on most RunPod GPUs)
- RAM: 16GB+ system RAM
- Disk: 5GB for model weights

**Inference Speed:**
- With 4-bit quantization: ~20-30 tokens/second
- Without quantization: ~10-15 tokens/second

---

## 🚀 Quick Commands Reference

```bash
# Set HuggingFace token (REQUIRED FIRST TIME)
export HF_TOKEN="hf_YOUR_TOKEN_HERE"

# Start server (background, persistent)
nohup python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8888 --host 0.0.0.0 --trust-remote-code \
  --max-model-len 4096 > /workspace/llm_server.log 2>&1 &
echo $! > /workspace/llm_server.pid

# Check if running
ps aux | grep vllm

# Check logs
tail -f /workspace/llm_server.log

# Monitor download progress
watch -n 5 'du -sh ~/.cache/huggingface/hub'

# Stop server
kill $(cat /workspace/llm_server.pid)

# Test locally (after model loads)
curl http://localhost:8888/health
curl http://localhost:8888/v1/models

# Test via RunPod proxy
curl https://ytc61lal7ag5sy-19123.proxy.runpod.net/2feph6uogs25wg1sc0i37280ah5ajfmm/v1/models
```

---

## ✅ Success Checklist

- [ ] HuggingFace token set in environment (`export HF_TOKEN=...`)
- [ ] Llama 3.1 terms accepted at https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
- [ ] vLLM installed (`pip install vllm`)
- [ ] Model download complete (~8GB, can take 5-10 minutes first time)
- [ ] Server started on port 8888
- [ ] "Application startup complete" message appears
- [ ] Health check returns "OK"
- [ ] Models endpoint returns Llama 3.1 8B
- [ ] Proxy URL works from your local machine
- [ ] Render backend LLM health check passes
- [ ] Chat endpoint returns real LLM responses (not fallback)
- [ ] Proxy URL accessible from outside
- [ ] Render LLM_API_URL has correct URL (with hyphen)
- [ ] Backend redeployed
- [ ] Backend LLM health shows "healthy"
- [ ] Chat API returns real responses

---

**You're almost there! Just start the server and fix the hyphen in Render!** 🚀

