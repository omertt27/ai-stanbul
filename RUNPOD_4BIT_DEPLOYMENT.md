# 🎯 Deploy 4-bit Llama 3.1 8B on RunPod (No Docker)

## Why 4-bit Quantization?

✅ **50% less VRAM** (~5GB instead of ~10GB)  
✅ **Faster inference** (50-80 tokens/s on A5000)  
✅ **Minimal quality loss** (<2% accuracy drop)  
✅ **Can run larger models** on same GPU  

---

## 🚀 Quick Deployment (5 minutes)

### Step 1: SSH into RunPod

```bash
ssh fgkqzve33ssbea-64411271@ssh.runpod.io -i ~/.ssh/id_ed25519
```

### Step 2: Install vLLM (one-time setup)

```bash
# Update pip
pip install --upgrade pip

# Install vLLM
pip install vllm

# Install/update PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Start 4-bit Llama (Multiple Options)

#### **Option A: AWQ 4-bit (Recommended - Fastest)**

```bash
python -m vllm.entrypoints.openai.api_server \
  --model TheBloke/Llama-3.1-8B-Instruct-AWQ \
  --quantization awq \
  --dtype float16 \
  --max-model-len 4096 \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code
```

#### **Option B: GPTQ 4-bit (Alternative)**

```bash
python -m vllm.entrypoints.openai.api_server \
  --model TheBloke/Llama-3.1-8B-Instruct-GPTQ \
  --quantization gptq \
  --dtype float16 \
  --max-model-len 4096 \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.85
```

#### **Option C: bitsandbytes 4-bit (Fallback)**

```bash
# If AWQ/GPTQ models aren't available, use regular model with bitsandbytes
pip install bitsandbytes

python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --load-in-4bit \
  --dtype float16 \
  --max-model-len 4096 \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.75
```

### Step 4: Run in Background (tmux)

```bash
# Start tmux session
tmux new -s llama

# Run the command from Step 3
python -m vllm.entrypoints.openai.api_server \
  --model TheBloke/Llama-3.1-8B-Instruct-AWQ \
  --quantization awq \
  --dtype float16 \
  --host 0.0.0.0 \
  --port 8000

# Detach from tmux: Press Ctrl+B, then D
# Reattach later: tmux attach -s llama
```

### Step 5: Test (on RunPod)

```bash
# Wait 2-3 minutes for model to load
# Check if running
curl http://localhost:8000/v1/models

# Test generation
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TheBloke/Llama-3.1-8B-Instruct-AWQ",
    "prompt": "What is Istanbul?",
    "max_tokens": 100
  }'
```

### Step 6: Get Your Public URL

1. Go to: https://www.runpod.io/console/pods
2. Find your pod: `fgkqzve33ssbea-64411271`
3. Look for **HTTP Service [8000]**
4. Copy the URL: `https://fgkqzve33ssbea-8000.proxy.runpod.net`

### Step 7: Configure AI Istanbul (on your local machine)

```bash
# Edit backend/.env
echo 'LLM_API_URL=https://fgkqzve33ssbea-8000.proxy.runpod.net/v1' >> backend/.env
echo 'LLM_TIMEOUT=60' >> backend/.env
echo 'LLM_MAX_TOKENS=250' >> backend/.env

# Test connection
cd backend
python test_runpod_connection.py

# Start backend
python api_server.py
```

---

## 🎯 Best 4-bit Models for Istanbul Tourism

### 1. **TheBloke/Llama-3.1-8B-Instruct-AWQ** ⭐ Recommended
```
--model TheBloke/Llama-3.1-8B-Instruct-AWQ
```
- **VRAM:** ~5GB
- **Speed:** 60-80 tokens/s
- **Quality:** Excellent

### 2. **TheBloke/Mistral-7B-Instruct-v0.3-AWQ** ⭐ Best for Turkish
```
--model TheBloke/Mistral-7B-Instruct-v0.3-AWQ
```
- **VRAM:** ~4.5GB
- **Speed:** 70-90 tokens/s
- **Turkish:** Better multilingual support

### 3. **TheBloke/Phi-3-mini-4k-instruct-AWQ** ⭐ Fastest
```
--model TheBloke/Phi-3-mini-4k-instruct-AWQ
```
- **VRAM:** ~3GB
- **Speed:** 100-120 tokens/s
- **Quality:** Good

---

## 📊 4-bit vs Full Precision Comparison

| Metric | Full (FP16) | 4-bit (AWQ) |
|--------|-------------|-------------|
| VRAM | ~10GB | ~5GB |
| Load Time | 30-60s | 15-30s |
| Tokens/sec | 40-60 | 60-80 |
| Quality Loss | 0% | <2% |

**Verdict:** Use 4-bit! 🚀

---

## 🐛 Troubleshooting

### Issue: "Model not found"

**Solution:** Use TheBloke's quantized models:
```bash
# Instead of:
--model meta-llama/Llama-3.1-8B-Instruct

# Use:
--model TheBloke/Llama-3.1-8B-Instruct-AWQ
```

### Issue: "torch not found"

**Solution:** Install PyTorch:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Issue: "CUDA out of memory"

**Solution:** Reduce GPU memory utilization:
```bash
--gpu-memory-utilization 0.7  # Instead of 0.85
```

### Issue: "Module vllm not found"

**Solution:** Install vLLM:
```bash
pip install vllm
```

### Issue: Process stopped after SSH disconnect

**Solution:** Use tmux or nohup:
```bash
# Option 1: tmux
tmux new -s llama
python -m vllm.entrypoints.openai.api_server ...
# Ctrl+B, then D to detach

# Option 2: nohup
nohup python -m vllm.entrypoints.openai.api_server ... > vllm.log 2>&1 &

# Check logs
tail -f vllm.log
```

---

## 📈 Expected Performance (4-bit on RTX A5000)

| Model | VRAM | Tokens/sec | Concurrent Users |
|-------|------|------------|------------------|
| Llama 3.1 8B AWQ | ~5GB | 60-80 | 20-40 |
| Mistral 7B AWQ | ~4.5GB | 70-90 | 30-50 |
| Phi-3 Mini AWQ | ~3GB | 100-120 | 40-60 |

---

## 🚦 Complete Command (Copy & Paste)

**For Llama 3.1 8B (4-bit):**

```bash
# On RunPod, inside tmux
tmux new -s llama

python -m vllm.entrypoints.openai.api_server \
  --model TheBloke/Llama-3.1-8B-Instruct-AWQ \
  --quantization awq \
  --dtype float16 \
  --max-model-len 4096 \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --enable-prefix-caching

# Detach: Ctrl+B, then D
# Exit SSH, server keeps running!
```

---

## ✅ Verification Checklist

- [ ] SSH into RunPod
- [ ] Installed vLLM and PyTorch
- [ ] Started vLLM server (in tmux)
- [ ] Tested locally: `curl http://localhost:8000/v1/models`
- [ ] Got RunPod proxy URL
- [ ] Updated `backend/.env` with URL
- [ ] Tested connection: `python test_runpod_connection.py`
- [ ] Started backend: `python api_server.py`
- [ ] Verified in admin dashboard

---

**Your 4-bit Llama is ready! 2x faster, 50% less memory! 🚀**
