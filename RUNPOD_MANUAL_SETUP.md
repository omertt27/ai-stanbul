# 🚀 Manual RunPod Setup Guide - Llama 3.1 8B (4-bit)

**Date:** November 20, 2025  
**Goal:** Set up vLLM with Llama 3.1 8B (4-bit quantization) on RunPod

---

## 📋 Prerequisites

1. **HuggingFace Account:** https://huggingface.co/join
2. **HuggingFace Token:** https://huggingface.co/settings/tokens
   - Create token with "Read access to contents of all public gated repos"
3. **Llama License:** https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
   - Click "Request Access" and accept license
4. **RunPod Connection:** SSH key at `~/.ssh/id_ed25519`

---

## Step 1: Connect to RunPod

```bash
# Connect via SSH
ssh root@194.68.245.173 -p 22186 -i ~/.ssh/id_ed25519
```

**Expected:** You should see a RunPod terminal prompt.

---

## Step 2: Install Dependencies

Copy and paste these commands one by one:

```bash
# Update pip
pip install --upgrade pip --break-system-packages

# Install HuggingFace Hub with CLI
pip install -U "huggingface_hub[cli]" --break-system-packages

# Install vLLM
pip install vllm --break-system-packages

# Install bitsandbytes for 4-bit quantization
pip install bitsandbytes --break-system-packages

# Install additional dependencies
pip install transformers accelerate --break-system-packages
```

**Time:** 3-5 minutes  
**Expected:** All packages install successfully with "Successfully installed" messages.

---

## Step 3: HuggingFace Authentication

```bash
# Login to HuggingFace
huggingface-cli login
```

**Prompt:** "Token (token will not be displayed):"  
**Action:** Paste your HuggingFace token (starts with `hf_...`)

**Expected:** "Login successful"

**Troubleshooting:**
- If you don't have a token: https://huggingface.co/settings/tokens
- If login fails: Make sure you've accepted the Llama 3.1 license

---

## Step 4: Download Llama 3.1 8B Model

Create and run the download script:

```bash
# Create download script
cat > /workspace/download_model.py << 'EOF'
from huggingface_hub import snapshot_download
import os

print("📥 Downloading Llama 3.1 8B Instruct...")
print("This may take 5-10 minutes (~4-5GB)")
print("")

try:
    model_path = snapshot_download(
        repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        cache_dir="/workspace/.cache/huggingface/hub",
        resume_download=True,
        local_files_only=False
    )
    
    print("")
    print("✅ Model downloaded successfully!")
    print(f"📁 Path: {model_path}")
    print("")
    
    # Show file sizes
    print("📋 Downloaded files:")
    for root, dirs, files in os.walk(model_path):
        for file in files:
            filepath = os.path.join(root, file)
            size = os.path.getsize(filepath) / (1024**3)
            print(f"   {file}: {size:.2f} GB")
    
    print("")
    print("Ready to start vLLM!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("")
    print("Common fixes:")
    print("1. Accept Llama license: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct")
    print("2. Check token permissions: https://huggingface.co/settings/tokens")
    print("3. Run: huggingface-cli login (and re-enter token)")
    exit(1)
EOF

# Run the download
python /workspace/download_model.py
```

**Time:** 5-10 minutes  
**Expected:** Progress bars showing download, then "✅ Model downloaded successfully!"

**Common Errors:**
- **401 Unauthorized:** Run `huggingface-cli login` again
- **Gated repo:** Accept license at https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
- **Network error:** Check internet connection, script will resume on retry

---

## Step 5: Verify Installation

```bash
# Check GPU
nvidia-smi

# Check model files
ls -lh /workspace/.cache/huggingface/hub/ | grep llama

# Check installed packages
pip list | grep -E "(vllm|transformers|bitsandbytes)"
```

**Expected:**
- GPU shows available (e.g., RTX A5000)
- Directory exists: `models--meta-llama--Meta-Llama-3.1-8B-Instruct`
- All packages listed with versions

---

## Step 6: Start vLLM Server (4-bit Quantized)

```bash
# Go to workspace
cd /workspace

# Kill any old processes
pkill -f vllm

# Start vLLM with 4-bit quantization
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --dtype half \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  > vllm.log 2>&1 &

# Note the process ID
echo "vLLM PID: $!"

# Wait a moment
sleep 10

# Watch the log
tail -f vllm.log
```

**Time:** 2-3 minutes to load model  
**Expected:** See messages like:
```
INFO:     Loading model...
INFO:     Model loaded in XX.XX seconds
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**To stop watching log:** Press `Ctrl+C` (server keeps running in background)

**Troubleshooting:**
- **OOM (Out of Memory):** Reduce `--gpu-memory-utilization` to 0.75 or 0.65
- **Model not found:** Re-run download script
- **Quantization error:** Try without `--quantization` (slower but more compatible)

---

## Step 7: Test vLLM (On RunPod)

```bash
# Test models endpoint
curl http://localhost:8000/v1/models

# Test chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "messages": [
      {"role": "user", "content": "Say hello in Turkish"}
    ],
    "max_tokens": 50
  }'
```

**Expected:**
- Models endpoint: JSON with model name
- Chat completion: Turkish greeting response

---

## Step 8: Set Up SSH Tunnel (On Your Mac)

Open a **new terminal on your Mac** (not RunPod):

```bash
cd /Users/omer/Desktop/ai-stanbul

# Update tunnel script with new connection details
./start_runpod_tunnel.sh
```

**Expected:** "✅ Tunnel active! You can now use http://localhost:8000"

**Keep this terminal open!**

---

## Step 9: Test from Your Mac

Open **another terminal on your Mac**:

```bash
# Test vLLM through tunnel
curl http://localhost:8000/v1/models

# Test chat
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 30
  }'
```

**Expected:** Same responses as Step 7

---

## Step 10: Start Backend (On Your Mac)

```bash
cd /Users/omer/Desktop/ai-stanbul/backend
source venv/bin/activate
python main_pure_llm.py
```

**Expected:** Backend starts on port 8002

**Check:** Browser http://localhost:8002/health should show "healthy"

---

## Step 11: Run Multi-Language Tests

In another Mac terminal:

```bash
cd /Users/omer/Desktop/ai-stanbul
source backend/venv/bin/activate
python test_multilanguage.py
```

**Expected:** Tests pass with 90%+ success rate across all 6 languages

---

## 🎯 Quick Reference

### Check vLLM Status (RunPod)
```bash
# Check if running
ps aux | grep vllm | grep -v grep

# Check log
tail -f /workspace/vllm.log

# Restart vLLM
pkill -f vllm
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
  --trust-remote-code \
  > vllm.log 2>&1 &
```

### SSH Connection Details
```bash
Host: 194.68.245.173
Port: 22186
User: root
Key: ~/.ssh/id_ed25519

# Connect:
ssh root@194.68.245.173 -p 22186 -i ~/.ssh/id_ed25519
```

### URLs
- **HuggingFace Tokens:** https://huggingface.co/settings/tokens
- **Llama License:** https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
- **vLLM (via tunnel):** http://localhost:8000/v1
- **Backend:** http://localhost:8002

---

## ⚠️ Common Issues

### "401 Unauthorized"
**Solution:** Re-run `huggingface-cli login` with your token

### "Gated repo" Error
**Solution:** Accept license at https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct

### vLLM Crashes
**Solution:** Check `tail -f /workspace/vllm.log` for errors, reduce GPU memory

### Tunnel Not Working
**Solution:** Check RunPod pod is running, verify SSH connection

### Backend Can't Reach vLLM
**Solution:** Ensure tunnel is running, check `LLM_API_URL` in `.env`

---

## ✅ Success Checklist

- [ ] Connected to RunPod via SSH
- [ ] Installed all dependencies
- [ ] Authenticated with HuggingFace
- [ ] Downloaded Llama 3.1 8B model
- [ ] Started vLLM server
- [ ] Tested vLLM on RunPod
- [ ] Started SSH tunnel on Mac
- [ ] Tested vLLM from Mac
- [ ] Started backend on Mac
- [ ] Ran multi-language tests
- [ ] All 6 languages working

---

**Last Updated:** November 20, 2025
