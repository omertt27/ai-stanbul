# vLLM Setup on RunPod - Quick Start

**✅ HuggingFace authentication is working!**

Now let's install vLLM and start the server with Llama 3.1 8B (4-bit AWQ).

---

## 🚀 **Step-by-Step Setup (Run in RunPod SSH)**

### **Step 1: Set Environment Variables**

```bash
# Set cache directory to /workspace to avoid disk quota issues
export HF_HOME=/workspace/.cache
export HF_TOKEN=AISTANBUL

# Verify they're set
echo "HF_HOME: $HF_HOME"
echo "HF_TOKEN: $HF_TOKEN"
```

---

### **Step 2: Install vLLM**

```bash
# Install vLLM (this may take a few minutes)
pip install vllm

# Verify installation
pip show vllm
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
```

**Expected output:**
```
Name: vllm
Version: 0.11.2
...
```

---

### **Step 3: Download Llama 3.1 8B AWQ Model**

**Option A: Direct Download (Recommended)**

```bash
# Download the AWQ quantized model (4-bit, much smaller!)
huggingface-cli download casperhansen/llama-3.1-8b-instruct-awq \
  --local-dir /workspace/llama-3.1-8b-awq \
  --cache-dir /workspace/.cache

# Check download progress
ls -lh /workspace/llama-3.1-8b-awq/
du -sh /workspace/llama-3.1-8b-awq/
```

**Option B: Use Python to Download**

```bash
python3 << 'EOF'
from huggingface_hub import snapshot_download

print("Downloading Llama 3.1 8B AWQ...")
snapshot_download(
    repo_id="casperhansen/llama-3.1-8b-instruct-awq",
    local_dir="/workspace/llama-3.1-8b-awq",
    cache_dir="/workspace/.cache"
)
print("Download complete!")
EOF
```

**Expected size:** ~5-6 GB (much smaller than 15GB full model!)

---

### **Step 4: Start vLLM Server**

```bash
# Start vLLM with AWQ quantization
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/llama-3.1-8b-awq \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --port 8888 \
  --host 0.0.0.0
```

**What you should see:**

```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8888 (Press CTRL+C to quit)
```

**⚠️ Keep this terminal open!** vLLM is now running.

---

### **Step 5: Test Locally (Open New SSH Session)**

In a **NEW terminal/SSH session** to your RunPod pod:

```bash
# Test health endpoint
curl http://localhost:8888/health

# Expected: {"status":"ok"}

# Test v1/models endpoint
curl http://localhost:8888/v1/models

# Test chat completion
curl -X POST http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/llama-3.1-8b-awq",
    "messages": [
      {"role": "user", "content": "What is Istanbul?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

**If you see JSON responses with generated text, vLLM is working! ✅**

---

## 🌐 **Step 6: Get Public Endpoint**

Your vLLM server is accessible via RunPod's public endpoint.

### **Get the URL:**

1. Go to: https://www.runpod.io/console/pods
2. Click on your pod: `nnqisfv2zk46t2`
3. Go to **"Connect"** tab
4. Look for **"HTTP Ports [8888]"** section
5. Copy the URL

**Format:**
```
https://nnqisfv2zk46t2-8888.proxy.runpod.net
```

**Your current endpoint should be:**
```
https://nnqisfv2zk46t2-19123.proxy.runpod.net/bg0emcs94jgdqec4ja8hgiqab8cafe9l/
```

**⚠️ Note:** The URL you provided has a path at the end. Let me know what the exact HTTP Port 8888 URL is from the RunPod console!

---

## 🧪 **Step 7: Test Public Endpoint (From Your Mac)**

Once you have the public URL, test it from your local machine:

```bash
# Replace with your actual endpoint!
export RUNPOD_ENDPOINT="https://nnqisfv2zk46t2-8888.proxy.runpod.net"

# Test health
curl $RUNPOD_ENDPOINT/health

# Test v1/models
curl $RUNPOD_ENDPOINT/v1/models

# Test chat completion
curl -X POST $RUNPOD_ENDPOINT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/llama-3.1-8b-awq",
    "messages": [
      {"role": "user", "content": "What are the top 3 places to visit in Istanbul?"}
    ],
    "max_tokens": 150,
    "temperature": 0.7
  }'
```

---

## 🎯 **Step 8: Update Backend .env**

Once the public endpoint is confirmed working, update your backend:

```bash
# On your Mac terminal:
cd /Users/omer/Desktop/ai-stanbul/backend

# Update .env file with new endpoint
# LLM_API_URL=https://nnqisfv2zk46t2-8888.proxy.runpod.net/v1

# Or let me know the exact endpoint and I'll update it for you!
```

---

## 🛡️ **Step 9: Keep vLLM Running (Background)**

Right now, vLLM stops if you close the terminal. Let's fix that:

### **Option A: Use Screen (Simple)**

```bash
# Kill current vLLM (Ctrl+C)

# Start in screen
screen -S vllm-server

# Inside screen, start vLLM:
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/llama-3.1-8b-awq \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --port 8888 \
  --host 0.0.0.0

# Wait for "Application startup complete"
# Then press: Ctrl+A, then D (to detach)

# To reattach: screen -r vllm-server
```

### **Option B: Run in Background with nohup**

```bash
nohup python -m vllm.entrypoints.openai.api_server \
  --model /workspace/llama-3.1-8b-awq \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --port 8888 \
  --host 0.0.0.0 \
  > /workspace/vllm.log 2>&1 &

# Check logs
tail -f /workspace/vllm.log

# Check if running
ps aux | grep vllm
```

---

## 🔍 **Troubleshooting**

### **Issue: "No module named 'vllm'"**

```bash
pip install vllm
python -c "import vllm"  # Should not error
```

### **Issue: "CUDA out of memory"**

```bash
# Reduce GPU memory utilization
--gpu-memory-utilization 0.7  # Instead of 0.85

# Or reduce max model length
--max-model-len 2048  # Instead of 4096
```

### **Issue: "Model not found"**

```bash
# Verify model files exist
ls -lh /workspace/llama-3.1-8b-awq/

# Should show: config.json, tokenizer.json, *.safetensors files

# If missing, re-download:
huggingface-cli download casperhansen/llama-3.1-8b-instruct-awq \
  --local-dir /workspace/llama-3.1-8b-awq
```

### **Issue: "Address already in use (port 8888)"**

```bash
# Find what's using port 8888
lsof -i :8888

# Kill it
kill -9 <PID>

# Or use different port
--port 8889  # Then update public endpoint URL
```

---

## ✅ **Success Checklist**

- [ ] HuggingFace authentication working (`hf whoami` shows username)
- [ ] vLLM installed (`pip show vllm` works)
- [ ] Model downloaded (~5-6 GB in `/workspace/llama-3.1-8b-awq/`)
- [ ] vLLM server started ("Application startup complete")
- [ ] Local health check works (`curl http://localhost:8888/health`)
- [ ] Local chat completion works
- [ ] Got public endpoint URL from RunPod console
- [ ] Public endpoint accessible from Mac
- [ ] vLLM running in background (screen or nohup)
- [ ] Backend `.env` updated with new endpoint
- [ ] Backend can call LLM API successfully

---

## 📊 **Monitor GPU Usage**

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Should show vLLM using GPU memory (around 8-10 GB)
```

---

## 🎉 **Next Steps**

Once vLLM is running and accessible:

1. **Copy your public endpoint**
2. **Test it from your Mac** (curl commands above)
3. **Let me know the endpoint** and I'll update your backend `.env`
4. **Test backend integration**
5. **Test frontend** - it should now get real AI responses!

---

**Need help?** See [RUNPOD_HF_TOKEN_FIX.md](./RUNPOD_HF_TOKEN_FIX.md) for authentication issues.
