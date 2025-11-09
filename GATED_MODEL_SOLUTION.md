# 🔓 Gated Model Solution - Open-Access Alternative

## Problem Identified ❌

The Llama 3.1 8B model is a **gated model** on Hugging Face that requires:
1. Meta's approval (request access at https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)
2. Hugging Face authentication token
3. Login via `huggingface-cli login`

**Error encountered:**
```
GatedRepoError: 401 Client Error
Cannot access gated repo for url https://huggingface.co/meta-llama/Meta-Llama-3.1-8B
Access to model meta-llama/Llama-3.1-8B is restricted.
```

---

## Solution Implemented ✅

Updated `llm_api_server.py` to use **open-access models** that don't require authentication.

### New Model Strategy

The server now tries models in this order:

1. **Mistral-7B-Instruct-v0.2** (7B, instruction-tuned, high quality)
   - Model: `mistralai/Mistral-7B-Instruct-v0.2`
   - Size: ~7B parameters
   - Best for: General conversation, instruction following
   
2. **Phi-2** (2.7B, Microsoft, strong performance)
   - Model: `microsoft/phi-2`
   - Size: ~2.7B parameters
   - Best for: Faster inference, good quality

3. **TinyLlama-1.1B-Chat** (1.1B, very fast)
   - Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
   - Size: ~1.1B parameters
   - Best for: Quick responses, lower memory

### Key Changes

1. **Fallback mechanism**: Automatically tries next model if one fails
2. **No authentication required**: All models are open-access
3. **Better error handling**: Logs which model is being attempted
4. **Memory efficient**: Smaller models work better on CPU

---

## 🚀 Next Steps on VM

The updated file is already on your VM. Now restart the server:

### Step 1: Stop Current Server (if running)

```bash
pkill -f llm_api_server
```

### Step 2: Start Updated Server

```bash
cd ~/ai-istanbul
./start_llm_server.sh
```

### Step 3: Monitor Startup

```bash
tail -f /tmp/llm_api_server.log
```

You should see:
```
📥 Attempting to load: mistralai/Mistral-7B-Instruct-v0.2
   Loading tokenizer...
   Loading model...
   ⚙️  Configuration: CPU-only, float32
✅ Successfully loaded: mistralai/Mistral-7B-Instruct-v0.2
```

### Step 4: Verify (from local machine)

```bash
curl http://35.210.251.24:8000/health
```

---

## 📊 Expected Startup Time

| Model | Download Size | Load Time | Total |
|-------|--------------|-----------|-------|
| Mistral-7B | ~14 GB | 2-3 min | 5-6 min |
| Phi-2 | ~5 GB | 1-2 min | 2-3 min |
| TinyLlama | ~2 GB | 30-60s | 1-2 min |

**First startup will download the model (~14GB for Mistral)**

---

## 🔄 Alternative: Use Llama with Authentication

If you want to use Llama 3.1 8B, you need to:

### 1. Request Access from Meta

Visit: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B
Click "Request Access" and wait for approval (usually instant)

### 2. Get Hugging Face Token

1. Go to: https://huggingface.co/settings/tokens
2. Create a new token with "Read" permissions
3. Copy the token

### 3. Login on VM

```bash
# SSH into VM
gcloud compute ssh instance-20251109-085407 --zone=europe-west1-b

# Activate venv
cd ~/ai-istanbul
source venv/bin/activate

# Install huggingface-hub CLI
pip install huggingface-hub

# Login (paste your token when prompted)
huggingface-cli login
```

### 4. Update Model in Code

Edit `llm_api_server.py` to prioritize Llama:

```python
MODEL_OPTIONS = [
    "meta-llama/Meta-Llama-3.1-8B",  # Now first priority
    "mistralai/Mistral-7B-Instruct-v0.2",
    "microsoft/phi-2",
]
```

---

## 💡 Recommendation

**For Production**: Use **Mistral-7B-Instruct-v0.2**
- ✅ No authentication required
- ✅ High quality instruction following
- ✅ Good performance on CPU
- ✅ 7B parameters (similar to Llama)
- ✅ Well-maintained by Mistral AI

**For Testing**: Use **Phi-2** or **TinyLlama**
- ✅ Faster startup
- ✅ Lower memory usage
- ✅ Good for development

---

## 🎯 Commands Ready to Use

Copy-paste these commands on the VM:

```bash
# Stop old server
pkill -f llm_api_server

# Start new server with open-access models
cd ~/ai-istanbul && ./start_llm_server.sh

# Monitor in another terminal
tail -f /tmp/llm_api_server.log

# Check memory usage
free -h

# Check CPU
htop
```

---

## ✅ Status

| Item | Status |
|------|--------|
| Problem Identified | ✅ Gated model requires auth |
| Solution Implemented | ✅ Open-access fallback models |
| File Updated | ✅ Uploaded to VM |
| Ready to Restart | ✅ Commands provided |

**Next:** Restart the server on the VM with the commands above! 🚀
