# 🔧 RunPod vLLM Troubleshooting Guide

**Date:** November 19, 2025  
**Issue:** vLLM authentication with Meta-Llama-3.1-8B-Instruct  
**Status:** Diagnostic and Resolution Guide

---

## 📋 Quick Reference: RunPod Connection

### New SSH Connection Details (Nov 19, 2025)
```bash
Host: ssh.runpod.io
Port: 13262
User: root
Key: ~/.ssh/id_ed25519

# Connect to RunPod:
ssh -i ~/.ssh/id_ed25519 root@ssh.runpod.io -p 13262

# Start SSH Tunnel (run from local Mac):
./start_runpod_tunnel.sh
```

---

## 🔍 Step 1: Diagnose Current State

### Check 1: Verify SSH Connection
```bash
# Test SSH connection
ssh -i ~/.ssh/id_ed25519 root@ssh.runpod.io -p 13262 "echo 'SSH Connection OK'"
```

**Expected:** "SSH Connection OK"  
**If fails:** Check SSH key permissions, RunPod pod status

### Check 2: Verify Model Cache
```bash
# SSH into RunPod and check model files
ssh -i ~/.ssh/id_ed25519 root@ssh.runpod.io -p 13262 << 'EOF'
echo "🔍 Checking model cache..."
ls -lh /workspace/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/
echo ""
echo "📦 Snapshot directory:"
ls -lh /workspace/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/
EOF
```

**Expected:** Directory with model files (config.json, model.safetensors, etc.)  
**If fails:** Model not downloaded - need to download first

### Check 3: Check HuggingFace Authentication Status
```bash
ssh -i ~/.ssh/id_ed25519 root@ssh.runpod.io -p 13262 << 'EOF'
echo "🔐 Checking HuggingFace authentication..."
if [ -f ~/.cache/huggingface/token ]; then
    echo "✅ HuggingFace token found"
    echo "Token (first 10 chars): $(head -c 10 ~/.cache/huggingface/token)..."
else
    echo "❌ No HuggingFace token found"
fi
EOF
```

**Expected:** Token found  
**If fails:** Need to login to HuggingFace

### Check 4: Check vLLM Process
```bash
ssh -i ~/.ssh/id_ed25519 root@ssh.runpod.io -p 13262 << 'EOF'
echo "🔍 Checking vLLM process..."
ps aux | grep vllm | grep -v grep
echo ""
echo "📋 vLLM log (last 30 lines):"
tail -n 30 /workspace/vllm.log
EOF
```

**Expected:** vLLM process running OR clear error in log  
**If fails:** vLLM not running or crashed

---

## 🔧 Step 2: Fix HuggingFace Authentication

### Option A: Login via CLI (Recommended)
```bash
# SSH into RunPod
ssh -i ~/.ssh/id_ed25519 root@ssh.runpod.io -p 13262

# Once connected:
# 1. Install/upgrade huggingface-hub
pip install --upgrade huggingface_hub --break-system-packages

# 2. Login to HuggingFace
huggingface-cli login

# 3. Enter your token when prompted
# Get token from: https://huggingface.co/settings/tokens
# Make sure token has "Read access to contents of all public gated repos" permission
```

### Option B: Login via Python Script (Alternative)
If CLI login fails, use Python script:

```bash
# On local Mac, create login script:
cat > /tmp/hf_login.py << 'EOF'
from huggingface_hub import login
import os

# Replace with your actual token
token = "hf_YOUR_TOKEN_HERE"
login(token=token)
print("✅ Logged in successfully!")
EOF

# Copy to RunPod and run:
scp -i ~/.ssh/id_ed25519 -P 13262 /tmp/hf_login.py root@ssh.runpod.io:/workspace/
ssh -i ~/.ssh/id_ed25519 root@ssh.runpod.io -p 13262 "cd /workspace && python hf_login.py"
```

### Option C: Set Environment Variable
```bash
# SSH into RunPod
ssh -i ~/.ssh/id_ed25519 root@ssh.runpod.io -p 13262

# Once connected:
export HF_TOKEN="hf_YOUR_TOKEN_HERE"
echo 'export HF_TOKEN="hf_YOUR_TOKEN_HERE"' >> ~/.bashrc
```

---

## 🚀 Step 3: Start vLLM Service

### Method 1: Standard Start (After Authentication)
```bash
# SSH into RunPod
ssh -i ~/.ssh/id_ed25519 root@ssh.runpod.io -p 13262

# Once connected, start vLLM:
cd /workspace

# Kill any existing vLLM processes
pkill -f vllm

# Start vLLM with standard model name (uses cache automatically)
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --dtype half \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 \
  --port 8000 \
  > vllm.log 2>&1 &

# Check log:
tail -f vllm.log
```

**Expected:** Model loads in 2-3 minutes, see "Uvicorn running on http://0.0.0.0:8000"

### Method 2: Direct Snapshot Path (If Method 1 Fails)
```bash
# Find exact snapshot path:
SNAPSHOT_PATH=$(ls -d /workspace/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/* | head -n 1)
echo "Using snapshot: $SNAPSHOT_PATH"

# Start vLLM with direct path:
python -m vllm.entrypoints.openai.api_server \
  --model $SNAPSHOT_PATH \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --dtype half \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  > vllm.log 2>&1 &
```

### Method 3: Offline Mode (If Network Issues)
```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --dtype half \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 \
  --port 8000 \
  > vllm.log 2>&1 &
```

---

## ✅ Step 4: Verify vLLM is Running

### From RunPod (SSH session):
```bash
# Check if service is responding:
curl http://localhost:8000/v1/models

# Expected: JSON response with model name
```

### From Local Mac (through tunnel):
```bash
# In a separate terminal, start tunnel:
./start_runpod_tunnel.sh

# Then test from local machine:
curl http://localhost:8000/v1/models

# Test chat completion:
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 50
  }'
```

---

## 🧪 Step 5: Test Multi-Language System

Once vLLM is running and tunnel is active:

```bash
# 1. Start backend (in one terminal):
cd /Users/omer/Desktop/ai-stanbul/backend
source venv/bin/activate
python main_pure_llm.py

# 2. Run multi-language tests (in another terminal):
cd /Users/omer/Desktop/ai-stanbul
source backend/venv/bin/activate
python test_multilanguage.py

# 3. Check results:
cat test_results_multilanguage_*.json
```

**Expected:** 90%+ pass rate with responses in correct languages

---

## 🐛 Common Issues and Solutions

### Issue 1: "401 Unauthorized" Error
**Symptom:** vLLM log shows "401 Client Error: Unauthorized"  
**Solution:** Login to HuggingFace (see Step 2)

### Issue 2: "Gated Repo" Error
**Symptom:** "You are trying to access a gated repo"  
**Solution:** 
1. Go to https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
2. Click "Request Access" and accept license
3. Wait for approval (usually instant)
4. Re-login to HuggingFace on RunPod

### Issue 3: vLLM Crashes During Startup
**Symptom:** Process exits with error  
**Solution:**
1. Check GPU memory: `nvidia-smi`
2. Reduce `--gpu-memory-utilization` to 0.75 or 0.65
3. Reduce `--max-model-len` to 1024
4. Check vllm.log for specific error

### Issue 4: SSH Connection Refused
**Symptom:** Cannot connect to RunPod  
**Solution:**
1. Check pod is running on RunPod dashboard
2. Verify SSH credentials are correct
3. Check if pod was restarted (SSH details may change)

### Issue 5: Tunnel Works but Backend Can't Reach vLLM
**Symptom:** Backend shows "Additional context temporarily unavailable"  
**Solution:**
1. Check LLM_API_URL in backend/.env: `http://localhost:8000/v1`
2. Verify tunnel is running: `lsof -i :8000`
3. Test endpoint manually: `curl http://localhost:8000/v1/models`

### Issue 6: Model Loads but Responses are Garbage
**Symptom:** vLLM responds but output is nonsense  
**Solution:**
1. Check quantization settings match model type
2. Verify correct model snapshot is loaded
3. Try without quantization for testing (slower but more reliable)

---

## 📊 Health Check Checklist

Use this checklist to verify full system health:

- [ ] SSH connection to RunPod works
- [ ] HuggingFace authentication configured
- [ ] Model cache exists on RunPod
- [ ] vLLM process running on RunPod
- [ ] vLLM endpoint responds to curl (from RunPod)
- [ ] SSH tunnel active on local Mac
- [ ] vLLM endpoint responds to curl (from Mac)
- [ ] Backend API running (port 8002)
- [ ] Backend can reach vLLM through tunnel
- [ ] Test queries return proper responses
- [ ] Multi-language tests pass (90%+ rate)

---

## 🔄 Quick Start from Scratch

If everything is broken and you need to start fresh:

```bash
# 1. Connect to RunPod and restart vLLM
ssh -i ~/.ssh/id_ed25519 root@ssh.runpod.io -p 13262 << 'EOF'
pkill -f vllm
huggingface-cli login  # Enter token
cd /workspace
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --dtype half \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 \
  --port 8000 \
  > vllm.log 2>&1 &
sleep 5
tail -n 50 vllm.log
EOF

# 2. Start tunnel (on Mac, in separate terminal)
cd /Users/omer/Desktop/ai-stanbul
./start_runpod_tunnel.sh

# 3. Test endpoint (on Mac, in another terminal)
curl http://localhost:8000/v1/models

# 4. Start backend (on Mac, in another terminal)
cd /Users/omer/Desktop/ai-stanbul/backend
source venv/bin/activate
python main_pure_llm.py

# 5. Run tests (on Mac, in another terminal)
cd /Users/omer/Desktop/ai-stanbul
source backend/venv/bin/activate
python test_multilanguage.py
```

---

## 📞 Next Steps

After vLLM is running successfully:

1. **Run comprehensive tests:** `python test_multilanguage.py`
2. **Verify all 6 languages:** Check test results for each language
3. **Test frontend integration:** Start frontend and test in browser
4. **Update documentation:** Update PHASE_3_CURRENT_STATUS.md with results
5. **Proceed to Phase 4:** Begin production deployment preparation

---

## 📝 Notes

- **Model Size:** ~15GB (8B parameters with quantization)
- **Startup Time:** 2-3 minutes for model loading
- **Memory:** Requires ~10GB GPU RAM with quantization
- **Token:** Get from https://huggingface.co/settings/tokens
- **Pod Type:** Minimum NVIDIA RTX 3090 or better recommended

**Last Updated:** November 19, 2025
