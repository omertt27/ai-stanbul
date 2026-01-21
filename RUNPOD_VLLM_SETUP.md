# vLLM Setup on RunPod - Complete Installation Guide

This guide will walk you through installing all dependencies, setting up HuggingFace authentication, downloading the model, and starting the vLLM server.

---

## 📋 **Quick Reference - Copy & Paste Commands**

For those who want to get started quickly, here's the full installation in one go:

```bash
# 1. Update system
apt-get update && apt-get install -y git wget curl vim screen tmux htop

# 2. Upgrade pip and install PyTorch
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install vLLM and HuggingFace tools
pip install vllm huggingface-hub transformers accelerate fastapi uvicorn pydantic

# 4. Set environment variables
export HF_HOME=/workspace/.cache
export TRANSFORMERS_CACHE=/workspace/.cache
mkdir -p /workspace/.cache /workspace/models
echo 'export HF_HOME=/workspace/.cache' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE=/workspace/.cache' >> ~/.bashrc

# 5. Login to HuggingFace (you'll need to paste your token)
huggingface-cli login

# 6. Download model (takes 10-20 minutes)
huggingface-cli download hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --local-dir /workspace/models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --cache-dir /workspace/.cache

# 7. Start vLLM server in screen
screen -S vllm-server
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --port 8000 \
  --host 0.0.0.0
# Press Ctrl+A then D to detach from screen
```

---

## 🚀 **Detailed Step-by-Step Setup**

Follow these steps if you want detailed explanations and troubleshooting tips.

## 🚀 **Step-by-Step Setup (Run in RunPod SSH)**

### **Step 1: Update System and Install Base Dependencies**

```bash
# Update package lists
apt-get update

# Install essential tools
apt-get install -y git wget curl vim screen tmux htop

# Verify Python version (should be 3.10+)
python --version
```

**Expected output:** `Python 3.10.x` or higher

---

### **Step 2: Install Python Dependencies**

```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

**Expected output:**
```
CUDA available: True
CUDA version: 12.1
GPU count: 1 (or more)
```

---

### **Step 3: Install vLLM and Dependencies**

```bash
# Install vLLM (this may take 5-10 minutes)
pip install vllm

# Install HuggingFace libraries
pip install huggingface-hub transformers accelerate

# Install additional utilities
pip install fastapi uvicorn pydantic

# Verify installations
pip show vllm
pip show huggingface-hub
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
```

**Expected output:**
```
Name: vllm
Version: 0.6.3 (or higher)
...
```

---

### **Step 4: Set Environment Variables**

```bash
# Set cache directory to /workspace to avoid disk quota issues
export HF_HOME=/workspace/.cache
export TRANSFORMERS_CACHE=/workspace/.cache
export HF_DATASETS_CACHE=/workspace/.cache

# Create cache directory
mkdir -p /workspace/.cache

# Add to .bashrc so it persists across sessions
echo 'export HF_HOME=/workspace/.cache' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE=/workspace/.cache' >> ~/.bashrc
echo 'export HF_DATASETS_CACHE=/workspace/.cache' >> ~/.bashrc

# Verify they're set
echo "HF_HOME: $HF_HOME"
echo "TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
```

---

### **Step 5: Login to HuggingFace**

```bash
# Install huggingface_hub CLI if not already installed
pip install -U "huggingface_hub[cli]"

# Login to HuggingFace (interactive)
huggingface-cli login

# You'll be prompted to enter your token
# Paste your HuggingFace token: hf_xxxxxxxxxxxxxxxxxxxxx
# Press Enter

# Verify login
huggingface-cli whoami
```

**Expected output:**
```
username: <your-username>
email: <your-email>
```

**Alternative: Set token without interactive prompt**

```bash
# Set token directly
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxx"  # Replace with your actual token

# Add to .bashrc
echo 'export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxx"' >> ~/.bashrc

# Login with token
huggingface-cli login --token $HF_TOKEN

# Verify
huggingface-cli whoami
```

**How to get your HuggingFace token:**
1. Go to: https://huggingface.co/settings/tokens
2. Click "New token"
3. Name it "runpod-vllm"
4. Select "Read" permissions
5. Copy the token (starts with `hf_`)

---

### **Step 6: Accept Llama 3.1 Model License**

Before downloading, you need to accept Meta's license:

1. Go to: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
2. Click "Agree and access repository"
3. Fill out the form and submit

**Wait for approval (usually instant to 1 hour)**

---

### **Step 7: Verify GPU and Disk Space**

```bash
# Check GPU
nvidia-smi

# Check disk space (you need at least 10GB free)
df -h /workspace

# Check available memory
free -h
```

---

### **Step 8: Download Llama 3.1 8B Model (4-bit AWQ)**

We'll use the **official Meta Llama 3.1 8B** model quantized to **4-bit AWQ** format for optimal performance.

**Model:** `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`
- ✅ Official Meta architecture
- ✅ 4-bit quantization (uses ~5GB instead of 16GB)
- ✅ AWQ format (optimized for vLLM)
- ✅ No accuracy loss for chat tasks

```bash
# Create model directory
mkdir -p /workspace/models

# Download the model (this will take 10-20 minutes depending on connection)
huggingface-cli download hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --local-dir /workspace/models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --cache-dir /workspace/.cache

# Monitor download progress
# You can open another terminal and run:
# watch -n 5 'du -sh /workspace/models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4'

# Verify download is complete
ls -lh /workspace/models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4/

# Check total size (should be around 5GB)
du -sh /workspace/models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4/
```

**Expected files:**
```
config.json
generation_config.json
tokenizer.json
tokenizer_config.json
model-00001-of-00002.safetensors
model-00002-of-00002.safetensors
```

**Alternative: Download using Python**

```bash
python3 << 'EOF'
from huggingface_hub import snapshot_download
import os

print("📥 Downloading Meta Llama 3.1 8B (4-bit AWQ)...")
print("This will take 10-20 minutes...")

snapshot_download(
    repo_id="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    local_dir="/workspace/models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    cache_dir="/workspace/.cache"
)

print("✅ Download complete!")
print("\nModel location: /workspace/models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4")
EOF
```

---

### **Step 9: Start vLLM Server**

Now that everything is installed, let's start the vLLM server with OpenAI-compatible API:

```bash
# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --port 8000 \
  --host 0.0.0.0
```

**What you should see:**

```
INFO 01-21 13:45:00 api_server.py:123] vLLM API server version 0.6.3
INFO 01-21 13:45:00 api_server.py:124] args: Namespace(model='/workspace/models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4', ...)
INFO 01-21 13:45:05 llm_engine.py:98] Initializing an LLM engine with config: ...
INFO 01-21 13:45:10 model_runner.py:140] Loading model weights took 4.5 GB
INFO 01-21 13:45:12 gpu_executor.py:76] # GPU blocks: 2048, # CPU blocks: 512
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**⚠️ Keep this terminal open!** vLLM is now running.

**Loading time:** 1-3 minutes for first startup

---

### **Step 10: Test Locally (Open New SSH Session)**

In a **NEW terminal/SSH session** to your RunPod pod:

```bash
# Test health endpoint
curl http://localhost:8000/health
```

**Expected:**
```json
{"status": "ok"}
```

```bash
# Test v1/models endpoint
curl http://localhost:8000/v1/models
```

**Expected:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "/workspace/models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
      "object": "model",
      ...
    }
  ]
}
```

```bash
# Test chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    "messages": [
      {"role": "user", "content": "What is Istanbul?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

**Expected:**
```json
{
  "id": "cmpl-xxx",
  "object": "chat.completion",
  "created": 1705847123,
  "model": "/workspace/models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Istanbul is a city in Turkey that straddles Europe and Asia..."
      },
      "finish_reason": "stop"
    }
  ],
  ...
}
```

**If you see JSON responses with generated text, vLLM is working! ✅**

---

## 🌐 **Step 11: Get Public Endpoint**

Your vLLM server is accessible via RunPod's public endpoint.

### **Get the URL:**

1. Go to: https://www.runpod.io/console/pods
2. Click on your pod
3. Go to **"Connect"** tab
4. Look for **"HTTP Ports [8000]"** section (or TCP Port Mappings)
5. Copy the public URL

**Format:**
```
https://<pod-id>-8000.proxy.runpod.net
```

**Example:**
```
https://jdstm70qd70cbd-8000.proxy.runpod.net
```

---

## 🧪 **Step 12: Test Public Endpoint (From Your Mac)**

Once you have the public URL, test it from your local machine (Mac terminal):

```bash
# Replace with your actual endpoint!
export RUNPOD_ENDPOINT="https://jdstm70qd70cbd-8000.proxy.runpod.net"

# Test health
curl $RUNPOD_ENDPOINT/health

# Test v1/models
curl $RUNPOD_ENDPOINT/v1/models

# Test chat completion
curl -X POST $RUNPOD_ENDPOINT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    "messages": [
      {"role": "user", "content": "What are the top 3 places to visit in Istanbul?"}
    ],
    "max_tokens": 150,
    "temperature": 0.7
  }'
```

**If this works, your vLLM server is publicly accessible! ✅**

---

## 🎯 **Step 13: Update Backend Configuration**

Once the public endpoint is confirmed working, let's update your backend to use it:

```bash
# On your Mac terminal:
cd /Users/omer/Desktop/ai-stanbul/backend

# Check current .env file
cat .env | grep -i runpod

# You should see something like:
# RUNPOD_ENDPOINT_URL=https://jdstm70qd70cbd-8000.proxy.runpod.net/v1
```

I'll help you update the `.env` file with the correct endpoint once you provide the public URL from RunPod.

---

## 🛡️ **Step 14: Keep vLLM Running in Background**

Right now, vLLM stops if you close the terminal. Let's fix that:

### **Option A: Use Screen (Recommended for RunPod)**

```bash
# If vLLM is currently running, stop it: Ctrl+C

# Start in screen
screen -S vllm-server

# Inside screen, start vLLM:
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --port 8000 \
  --host 0.0.0.0

# Wait for "Application startup complete"
# Then press: Ctrl+A, then D (to detach)

# To check if it's running:
screen -ls

# To reattach later:
screen -r vllm-server

# To kill the screen session:
screen -X -S vllm-server quit
```

### **Option B: Run in Background with nohup**

```bash
# Stop vLLM if running (Ctrl+C)

# Start in background
nohup python -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --port 8000 \
  --host 0.0.0.0 \
  > /workspace/vllm.log 2>&1 &

# Check logs
tail -f /workspace/vllm.log

# Press Ctrl+C to exit log view (server keeps running)

# Check if running
ps aux | grep vllm
netstat -tlnp | grep 8000
```

### **Option C: Create a Startup Script**

```bash
# Create a startup script
cat > /workspace/start_vllm.sh << 'EOF'
#!/bin/bash
echo "Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --port 8000 \
  --host 0.0.0.0
EOF

# Make it executable
chmod +x /workspace/start_vllm.sh

# Run it
/workspace/start_vllm.sh
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
ls -lh /workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4/

# Should show: config.json, tokenizer.json, *.safetensors files

# If missing, re-download:
huggingface-cli download hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --local-dir /workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
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
- [ ] Official Meta Llama model downloaded (~5GB in `/workspace/Meta-Llama-3.1-8B-Instruct-AWQ-INT4/`)
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
